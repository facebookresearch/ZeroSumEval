import json
import argparse
from datasets import load_dataset
from pathlib import Path

def download_gsm8k():
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset["train"]
    output_file = Path(__file__).parent / "gsm8k_train.jsonl"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for example in train_data:
            json.dump(example, f)
            f.write("\n")
    
    print(f"GSM8K training set saved to {output_file}")

def download_other(dataset):
    dataset = load_dataset(dataset)
    train_data = dataset["train"]
    output_file = Path(__file__).parent / f"{dataset}_train.jsonl"
    
    with open(output_file, "w", encoding="utf-8") as f:
        for example in train_data:
            json.dump(example, f)
            f.write("\n")
    
    print(f"{dataset} training set saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GSM8K or MATH dataset")
    parser.add_argument("dataset", choices=["gsm8k", "other"], help="Choose which dataset to download: 'gsm8k' or 'other'")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset to download", required=False)
    args = parser.parse_args()

    if args.dataset == "gsm8k":
        download_gsm8k()
    elif args.dataset == "other":
        download_other(args.dataset_name)
