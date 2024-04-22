from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


NUM_CLASSES = 3

X_NAME = "Permuted Proportion"

CHANNEL_SIZE = 5


def plot_channel_subset_permutation_zsc(zs_coord_df, self_play_df):
    # set_plotting_style(font_scale=2.5, rc={"legend.fontsize": 15}, use_times_font=False)
    set_plotting_style(font_scale=2.5, rc={"legend.fontsize": 15})

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        x=[-20, 20],
        y=[1 / NUM_CLASSES, 1 / NUM_CLASSES],
        color=(0.1, 0.1, 0.1, 0.5),
        label="Baseline",
    )
    ax.lines[0].set_linestyle("--")

    sns.lineplot(
        x=X_NAME,
        y="Zero-Shot Coordination Score",
        data=zs_coord_df,
        label="Zero-shot Performance",
    )
    sns.scatterplot(
        x=X_NAME, y="Zero-Shot Coordination Score", data=zs_coord_df, marker="x"
    )

    sns.lineplot(
        x=X_NAME,
        y="Self-play Performance",
        data=self_play_df,
        label="Self-play Performance",
    )
    # sns.scatterplot(x=X_NAME, y='Self-play Performance',
    #                 data=self_play_df, marker='x')

    sns.scatterplot(
        x="x",
        y="y",
        data=pd.DataFrame([{"x": 100, "y": 100}]),
        color=(0.1, 0.1, 0.1, 0.5),
        marker="x",
        label="Raw Data",
    )
    plt.ylim([-0.05, 1.05])
    plt.xlim([1.95, CHANNEL_SIZE + 0.05])
    # plt.title('The Effect of Channel Permutation on Zero-Shot Coordination')
    plt.ylabel("Performance")
    plt.xlabel("Subset Size")
    plt.xticks(pd.unique(zs_coord_df[X_NAME]))
    # plt.xlabel(X_NAME)

    plt.legend(loc=4)


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/channel_subset_permutations_zsc").glob(
            "data_*.csv"
        )
    ]
    plot_channel_subset_permutation_zsc(*data)
    plt.savefig(
        "figures/channel_subset_permutations_zsc/channel_subset_permutations_zsc.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
