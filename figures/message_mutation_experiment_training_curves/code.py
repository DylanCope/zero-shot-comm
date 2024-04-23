from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot_message_mutation_experiment_training_curves(history_df):
    set_plotting_style(font_scale=2.5)

    plt.figure(figsize=(12, 6))

    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)

    ax = sns.lineplot(
        history_df,
        x="epoch",
        y="ground_truth_acc",
        hue=r"$p_{m}$",
        palette="viridis",
        legend=False,
    )

    # ax.get_legend().remove()
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.9)
    cbar.set_label(
        r"Mutation Probability $p_{m}$",
        rotation=-90,
        labelpad=50.0,
    )

    ax.set_xticks([1] + list(range(10, 71, 10)))

    plt.xlabel("Training Epoch")
    plt.ylabel("Self-Play Test Accuracy")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path(
            "figures/message_mutation_experiment_training_curves"
        ).glob("data_*.csv")
    ]
    plot_message_mutation_experiment_training_curves(*data)
    plt.savefig(
        "figures/message_mutation_experiment_training_curves/message_mutation_experiment_training_curves.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
