# lora_train.py  ─ full script
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_NAME = "answerdotai/ModernBERT-base"
DATASET_PATH = "dataset.csv"
OUTPUT_DIR = "./results_lora"
LOGGING_DIR = "./logs_lora"

NUM_LABELS = 6  # 0-5
MAX_LENGTH = 128
TEST_SIZE = 0.20  # 20 % test
VALIDATION_SIZE = 0.125  # 10 % val of full dataset
SEED = 42

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["Wqkv"]  # ModernBERT combined QKV layer

# ─── Mixed-precision toggle ───────────────────────────────────────────────────
USE_CUDA = torch.cuda.is_available()
USE_FP16 = USE_CUDA  # safe on NVIDIA GPUs
USE_BF16 = (
    (not USE_CUDA) and torch.backends.mps.is_available() and hasattr(torch, "bf16")
)

print(f"CUDA available: {USE_CUDA}  |  fp16: {USE_FP16}  |  bf16: {USE_BF16}")

# ─── 1. Load dataset ──────────────────────────────────────────────────────────
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
df["label"] = df["label"].astype(int)
if df["label"].min() < 0 or df["label"].max() >= NUM_LABELS:
    raise ValueError(f"Label range should be 0-{NUM_LABELS - 1}")

dataset = Dataset.from_pandas(df)
print(f"Samples: {len(dataset)}")

# ─── 2. Train/Val/Test split ──────────────────────────────────────────────────
train_test_split = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
temp_train = train_test_split["train"]
test_dataset = train_test_split["test"]

train_valid_split = temp_train.train_test_split(test_size=VALIDATION_SIZE, seed=SEED)
train_dataset = train_valid_split["train"]
valid_dataset = train_valid_split["test"]

data_splits = DatasetDict(
    train=train_dataset,
    validation=valid_dataset,
    test=test_dataset,
)
print(
    f"Train {len(train_dataset)} | Val {len(valid_dataset)} | Test {len(test_dataset)}"
)

# ─── 3. Tokenizer & base model ────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)

# ─── 4. LoRA wrapper ──────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()


# ─── 5. Tokenise ──────────────────────────────────────────────────────────────
def tokenize(batch):
    if "text" not in batch:
        raise KeyError("'text' column missing")
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
    )


tokenised = data_splits.map(tokenize, batched=True)
tokenised.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ─── 6. TrainingArguments ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    weight_decay=0.01,
    logging_dir=LOGGING_DIR,
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="tensorboard",
    fp16=USE_FP16,
    bf16=USE_BF16,
    gradient_accumulation_steps=2,
    lr_scheduler_type="linear",
    warmup_ratio=0.06,
)

# ─── 7. Trainer ───────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenised["train"],
    eval_dataset=tokenised["validation"],
    tokenizer=tokenizer,  # still accepted; warning is fine
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# ─── 8. Train ─────────────────────────────────────────────────────────────────
train_result = trainer.train()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

# ─── 9. Save adapter ──────────────────────────────────────────────────────────
final_adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
model.save_pretrained(final_adapter_path)
tokenizer.save_pretrained(final_adapter_path)
print(f"LoRA adapter saved to {final_adapter_path}")

# ─── 10. Loss curve ───────────────────────────────────────────────────────────
logs = trainer.state.log_history
train_pairs = [(l["step"], l["loss"]) for l in logs if "loss" in l]
val_pairs = [(l["step"], l["eval_loss"]) for l in logs if "eval_loss" in l]

if train_pairs and val_pairs:
    tr_steps, tr_loss = zip(*train_pairs)
    va_steps, va_loss = zip(*val_pairs)

    plt.figure(figsize=(15, 6))
    plt.plot(tr_steps, tr_loss, label="Train Loss", alpha=0.7)
    plt.plot(va_steps, va_loss, label="Val Loss", linestyle="--", marker="o")
    plt.title("LoRA Training / Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("lora_loss_curve.png", dpi=300)
    print("Saved loss curve as lora_loss_curve.png")

# ─── 11. Test evaluation ──────────────────────────────────────────────────────
test_metrics = trainer.evaluate(eval_dataset=tokenised["test"])
trainer.log_metrics("test", test_metrics)
trainer.save_metrics("test", test_metrics)
print("Test metrics:", test_metrics)

print("Done.")
