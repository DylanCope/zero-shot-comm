from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
from seaborn.palettes import color_palette
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot_subset_permutation_experiment_training_curves(history_df):
    set_plotting_style(font_scale=2.5)

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        history_df,
        x="epoch",
        y="ground_truth_acc",
        hue="Subset Size",
        palette=color_palette("viridis", 5),
    )

    plt.xlabel("Training Epoch")
    plt.ylabel("Self-Play Test Accuracy")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path(
            "figures/subset_permutation_experiment_training_curves"
        ).glob("data_*.csv")
    ]
    plot_subset_permutation_experiment_training_curves(*data)
    plt.savefig(
        "figures/subset_permutation_experiment_training_curves/subset_permutation_experiment_training_curves.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
