import os
from dataloader import load_dataset
from utils import create_chunked_dataframe, compare_lengths, make_hf_datasets
from train import train_model
from utils import compute_metrics

def main():
    # 1. Paths (adjust if not on Kaggle)
    train_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train"
    label_path = "/kaggle/input/fake-or-real-the-impostor-hunt/data/train.csv"

    # 2. Load dataset
    df = load_dataset(train_path, label_path)
    print(f"Loaded dataset with {len(df)} pairs.")

    # 3. Exploratory stats
    compare_lengths(df)

    # 4. Preprocess into chunks
    chunked_df = create_chunked_dataframe(df)
    print(f"Created {len(chunked_df)} chunked samples.")

    # 5. HuggingFace dataset split
    train_dataset, val_dataset, test_dataset = make_hf_datasets(chunked_df)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 6. Train model
    trainer = train_model(train_dataset, val_dataset)

    # 7. Evaluate
    results = trainer.evaluate(test_dataset)
    print("Test Results:", results)

if __name__ == "__main__":
    main()