import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset.csv")  # Replace with your dataset path
dataset = Dataset.from_pandas(df)

# Split into train and test sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Load ModernBERT tokenizer and model with increased dropout
model_name = "answerdotai/ModernBERT-base"  # Replace with your specific model ID
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Increase dropout in the classifier (if supported by the model)
config = AutoConfig.from_pretrained(model_name, num_labels=6, classifier_dropout=0.3)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# Freeze initial layers to help reduce overfitting.
# Since ModernBERT might not have a 'bert' attribute, we try alternatives.
base_model = None
if hasattr(model, "bert"):
    base_model = model.bert
elif hasattr(model, "modernbert"):
    base_model = model.modernbert
elif hasattr(model, "base_model"):
    base_model = model.base_model
else:
    print(
        "Warning: Could not find the expected transformer module for freezing layers."
    )

if base_model is not None and hasattr(base_model, "embeddings"):
    for param in base_model.embeddings.parameters():
        param.requires_grad = False


# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch tensors
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define training arguments with a lower learning rate
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-6,  # Lower learning rate for more stable training
    lr_scheduler_type="linear",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    warmup_ratio=0.06,
)

# Initialize Trainer with an early stopping callback (patience=3)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train the model and capture training logs
train_output = trainer.train()

# Extract loss values from logs for plotting
training_logs = trainer.state.log_history  # Contains all logged metrics
train_losses = [log["loss"] for log in training_logs if "loss" in log]
val_losses = [log["eval_loss"] for log in training_logs if "eval_loss" in log]

# Plot training and validation loss curves
plt.figure(figsize=(20, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(
    range(1, len(val_losses) + 1), val_losses, label="Validation Loss", linestyle="--"
)
plt.title("Training and Validation Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
print("Loss curve saved as 'loss_curve.png'")
