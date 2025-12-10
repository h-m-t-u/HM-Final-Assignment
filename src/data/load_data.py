import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load the credit card transaction dataset."""
    return pd.read_csv(file_path)


if __name__ == "__main__":
    df = load_dataset("data/processed/train.csv")
    print(df.head())
