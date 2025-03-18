import pandas as pd
from transformers import (
    AutoTokenizer,
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

# Load ModernBERT tokenizer and model
model_name = (
    "answerdotai/ModernBERT-base"  # Replace with the specific ModernBERT model ID
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)


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

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=1e-5,
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

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
)

# Train the model and capture training logs
train_output = trainer.train()

# Extract loss values from logs for plotting
training_logs = trainer.state.log_history  # Contains all logged metrics

train_losses = [log["loss"] for log in training_logs if "loss" in log]
val_losses = [log["eval_loss"] for log in training_logs if "eval_loss" in log]

# Plot both training and validation loss curves and save as PNG
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

# Save the plot as a PNG file
plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
print("Loss curve saved as 'loss_curve.png'")
