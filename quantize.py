# quantize.py
import os
import argparse
import json
import pandas as pd
import torch
import quanto
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# Fix for tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants from the project
NUM_LABELS = 6
MAX_LENGTH = 128
TEST_SIZE = 0.20
SEED = 42


# --- Main Logic ---
def quantize_and_evaluate(
    model_path: str,
    output_dir: str,
    quant_level: str,
    dataset_path: str,
):
    """
    Loads a model, quantizes it using 'quanto', evaluates, and saves it.
    """
    print(f"--- Starting Quantization and Evaluation with `quanto` ---")
    print(f"Model Path: {model_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Quantization: {quant_level}")
    print(f"Dataset: {dataset_path}")

    # --- 1. Detect Device (MPS for Mac, CUDA, or CPU) ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch_dtype = torch.float32
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32
    print(f"\n--- Using device: {device} with dtype: {torch_dtype} ---")

    # --- 2. Load Tokenizer and Model ---
    print("\n--- Loading tokenizer and base model... ---")
    base_model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        torch_dtype=torch_dtype,
    ).to(device)
    print("Base model loaded successfully.")

    # --- 3. Apply Quantization with `quanto` ---
    print(f"\n--- Applying {quant_level} quantization... ---")
    q_map = {
        "int8": quanto.qint8,
        "float8": quanto.qfloat8,
        "int4": quanto.qint4,
        "int2": quanto.qint2,
    }
    if quant_level not in q_map:
        raise ValueError(f"quant_level must be one of {list(q_map.keys())}")

    # Use more balanced quantization - also quantize activations with less aggressive precision
    if quant_level in ["int4", "int2"]:
        # For aggressive weight quantization, use int8 activations for stability
        quanto.quantize(model, weights=q_map[quant_level], activations=quanto.qint8)
    else:
        # For int8/float8 weights, can use same precision for activations
        quanto.quantize(model, weights=q_map[quant_level], activations=q_map[quant_level])
    print("Model quantized. Freezing model for inference...")
    quanto.freeze(model)
    print("Model frozen successfully.")

    # --- 4. Load and Prepare Dataset for Evaluation ---
    print("\n--- Loading and preparing dataset for evaluation... ---")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    full_dataset = Dataset.from_pandas(df)
    train_test_split = full_dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    test_dataset = train_test_split["test"]

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    tokenized_test.set_format(
        type="torch", columns=["input_ids", "attention_mask", "label"]
    )
    print(f"Test dataset prepared with {len(tokenized_test)} samples.")

    # --- 5. Evaluate Model ---
    print("\n--- Evaluating quantized model... ---")
    eval_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "eval_temp"),
        per_device_eval_batch_size=8,
        report_to="none",
        use_mps_device=torch.backends.mps.is_available(),
    )
    trainer = Trainer(model=model, args=eval_args, eval_dataset=tokenized_test)
    metrics = trainer.evaluate()
    print("\n--- Evaluation Metrics ---")
    print(json.dumps(metrics, indent=2))

    # --- 6. Save Model and Metrics ---
    print(f"\n--- Saving quantized model to {output_dir}... ---")
    os.makedirs(output_dir, exist_ok=True)

    # ** THE FIX **
    # The standard `save_pretrained` can fail with frozen `quanto` models.
    # We save the config and tokenizer using the standard method, but save the
    # state_dict manually. This is a robust way to handle this issue.
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved. Metrics saved to {metrics_path}")
    print("\n--- Quantization and Evaluation Complete ---")


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantize and evaluate a Hugging Face model using `quanto`."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the quantized model and evaluation results.",
    )
    parser.add_argument(
        "--quant_level",
        type=str,
        required=True,
        choices=["int8", "float8", "int4", "int2"],
        help="Quantization level with `quanto`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset.csv",
        help="Path to the dataset CSV file.",
    )
    args = parser.parse_args()

    quantize_and_evaluate(
        model_path=args.model_path,
        output_dir=args.output_dir,
        quant_level=args.quant_level,
        dataset_path=args.dataset_path,
    )
