from src.data.load_data import load_dataset
from src.data.preprocess import clean_dataset
from src.visualization.eda import plot_eda
from src.models.train_model import split_data, plot_roc_curve
from src.models.knn_model import train_knn_model
from src.models.dumb_model import train_dumb_model
from src.visualization.performance import (
    plot_confusion_matrices,
    plot_performance_comparison,
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def main() -> None:
    print("---Loading data...")
    raw_df = load_dataset("data/train.csv")
print("The notebooks I used are in the Notebooks tab. I am submitting this, but I don't understand it.")



if __name__ == "__main__":
    main()
