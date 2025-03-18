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
import optuna  # Make sure you have installed optuna
import os

# ---------------------
# Load your dataset
# ---------------------
df = pd.read_csv("questions.csv")  # Replace with your dataset path
dataset = Dataset.from_pandas(df)

train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# ---------------------
# Define model/tokenizer names
# ---------------------
model_name = "answerdotai/ModernBERT-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)


# Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


# ---------------------
# model_init for re-initializing the model
# ---------------------
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# ---------------------
# Training Arguments
# ---------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    # We'll override these in the hyperparameter search,
    # but we need to put some defaults here anyway:
    learning_rate=1e-5,
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

# ---------------------
# Create the Trainer
# ---------------------
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
)


# ---------------------
# Define Hyperparameter Search Space
# ---------------------
def hp_space_optuna(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8, 16, 32]
        ),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01),
        # You could add more hyperparams to tune, e.g. "num_train_epochs", "warmup_ratio", etc.
    }


# ---------------------
# Run the Hyperparameter Search
# ---------------------
best_run = trainer.hyperparameter_search(
    direction="minimize",  # "minimize" if we want to minimize the validation loss
    hp_space=hp_space_optuna,  # Our function that tells optuna how to sample hyperparams
    n_trials=10,  # How many trials to run - increase for a more thorough search
)

print("Best hyperparameters found:", best_run.hyperparameters)

# ---------------------
# Re-train with Best Hyperparams
# ---------------------
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()

# ---------------------
# Plot Loss Curves (Optional)
# ---------------------
training_logs = trainer.state.log_history

train_losses = [log["loss"] for log in training_logs if "loss" in log]
val_losses = [log["eval_loss"] for log in training_logs if "eval_loss" in log]

plt.figure(figsize=(10, 5))
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
plt.show()
