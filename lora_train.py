# lora_train.py
import pandas as pd
import matplotlib.pyplot as plt
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict

# --- LoRA imports ---
from peft import LoraConfig, get_peft_model, TaskType

# --- Configuration ---
MODEL_NAME = "answerdotai/ModernBERT-base"
DATASET_PATH = "dataset.csv"  # Make sure this points to your dataset
OUTPUT_DIR = "./results_lora"  # Separate directory for LoRA results/checkpoints
LOGGING_DIR = "./logs_lora"
NUM_LABELS = 6  # Based on your dataset_example.csv (0-5)
MAX_LENGTH = 128
TEST_SIZE = 0.2  # 20% for test set
VALIDATION_SIZE = 0.125  # ~10% overall for validation (12.5% of the 80% training data)
SEED = 42

# --- LoRA Configuration ---
LORA_R = 16  # LoRA rank (can be tuned)
LORA_ALPHA = 32  # Scaling factor (often 2*r)
LORA_DROPOUT = 0.1
# Target the combined QKV layer found in ModernBERT
LORA_TARGET_MODULES = ["Wqkv"]

# 1. Load the dataset
print(f"Loading dataset from: {DATASET_PATH}")
if not os.path.exists(DATASET_PATH):
    print(f"Error: Dataset file not found at {DATASET_PATH}")
    exit(1)
try:
    df = pd.read_csv(DATASET_PATH)
    # Ensure 'label' column is integer type
    df["label"] = df["label"].astype(int)
    if df["label"].min() < 0 or df["label"].max() >= NUM_LABELS:
        print(f"Error: Labels in dataset should be between 0 and {NUM_LABELS - 1}.")
        print(f"Found min: {df['label'].min()}, max: {df['label'].max()}")
        exit(1)
    dataset = Dataset.from_pandas(df)
    print(f"Dataset loaded successfully with {len(dataset)} samples.")
except Exception as e:
    print(f"Error loading or processing dataset: {e}")
    exit(1)


# 2. Split the dataset (Train/Validation/Test)
print("Splitting dataset...")
# Split off test set first
train_test_split = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
temp_train_valid_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Split remaining into train and validation
train_valid_split = temp_train_valid_dataset.train_test_split(
    test_size=VALIDATION_SIZE, seed=SEED
)
train_dataset = train_valid_split["train"]
valid_dataset = train_valid_split["test"]

# Combine into a DatasetDict
split_datasets = DatasetDict(
    {"train": train_dataset, "validation": valid_dataset, "test": test_dataset}
)
print(
    f"Dataset split: Train={len(split_datasets['train'])}, Validation={len(split_datasets['validation'])}, Test={len(split_datasets['test'])}"
)


# 3. Load Tokenizer and Base Model
print(f"Loading tokenizer and base model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Load the base model *without* PEFT first
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    # Consider adding ignore_mismatched_sizes=True if loading a base checkpoint with a different head
)
print("Base model loaded.")


# 4. Set up LoRA configuration and apply PEFT
print("Setting up LoRA configuration...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",  # Common setting for LoRA
    task_type=TaskType.SEQ_CLS,  # Important for sequence classification
)
print(f"LoRA Config: {lora_config}")

# Wrap the base model with LoRA
model = get_peft_model(base_model, lora_config)
print("PEFT model created.")
model.print_trainable_parameters()  # Show how many parameters are trainable


# 5. Tokenize the datasets
print("Tokenizing datasets...")


def tokenize_function(examples):
    # Ensure 'text' column exists
    if "text" not in examples:
        raise KeyError("Dataset missing 'text' column for tokenization.")
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH
    )


try:
    tokenized_datasets = split_datasets.map(tokenize_function, batched=True)
    print("Tokenization complete.")
except KeyError as e:
    print(f"Error during tokenization: {e}")
    exit(1)


# 6. Set the format for PyTorch tensors
tokenized_datasets.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)
print("Dataset format set for PyTorch.")


# 7. Define Training Arguments
print("Defining training arguments...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",  # Evaluate every epoch
    save_strategy="epoch",  # Save checkpoint every epoch
    learning_rate=1e-4,  # LoRA often uses higher LR than full fine-tuning
    per_device_train_batch_size=16,  # Can often use larger batches with LoRA
    per_device_eval_batch_size=16,
    num_train_epochs=15,  # Adjust as needed
    weight_decay=0.01,
    logging_dir=LOGGING_DIR,
    logging_steps=20,
    load_best_model_at_end=True,  # Load the best model based on validation loss
    metric_for_best_model="loss",  # Can also use 'eval_loss' or other metrics
    greater_is_better=False,
    push_to_hub=False,  # Set to True if you want to push to Hugging Face Hub
    report_to="tensorboard",  # Or "wandb" if configured
    fp16=True,  # Use mixed precision if GPU supports it
    gradient_accumulation_steps=2,  # Accumulate gradients if batch size is effectively large
    lr_scheduler_type="linear",
    warmup_ratio=0.06,
)


# 8. Initialize Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,  # Use the PEFT model
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets[
        "validation"
    ],  # Use validation set for eval during training
    tokenizer=tokenizer,  # Pass tokenizer for potential padding/collating needs
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=5)
    ],  # Stop if no improvement for 5 epochs
)


# 9. Train the model
print("Starting LoRA training...")
train_result = trainer.train()
print("Training finished.")

# Save metrics and state
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
print("Training metrics and state saved.")

# Save the final adapter model explicitly (Trainer also saves checkpoints)
final_adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
model.save_pretrained(final_adapter_path)
tokenizer.save_pretrained(final_adapter_path)
print(f"Final LoRA adapter saved to: {final_adapter_path}")


# 10. Plot Loss Curve (Optional but helpful)
print("Plotting loss curve...")
try:
    training_logs = trainer.state.log_history
    train_steps_loss = [
        (log["step"], log["loss"]) for log in training_logs if "loss" in log
    ]
    eval_steps_loss = [
        (log["step"], log["eval_loss"]) for log in training_logs if "eval_loss" in log
    ]

    if train_steps_loss and eval_steps_loss:
        train_steps, train_losses = zip(*train_steps_loss)
        eval_steps, val_losses = zip(*eval_steps_loss)

        plt.figure(figsize=(15, 6))
        plt.plot(train_steps, train_losses, label="Train Loss", alpha=0.7)
        plt.plot(
            eval_steps, val_losses, label="Validation Loss", marker="o", linestyle="--"
        )

        plt.title("LoRA Training and Validation Loss Curve")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_curve_path = "lora_loss_curve.png"
        plt.savefig(loss_curve_path, dpi=300)
        print(f"Loss curve saved as '{loss_curve_path}'")
    else:
        print("Could not plot loss curve: Missing training or validation loss data.")

except Exception as e:
    print(f"Error plotting loss curve: {e}")


# 11. Final Evaluation on the Test Set
print("Evaluating model on the test set...")
if "test" in tokenized_datasets:
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    print("Test evaluation results:", test_results)
else:
    print("No test set found for final evaluation.")

print("LoRA training script finished.")
