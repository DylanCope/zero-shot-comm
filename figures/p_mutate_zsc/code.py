from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


NUM_CLASSES = 3


def create_message_mutation_zsc_plot(self_play_df, zs_coord_df):
    set_plotting_style(font_scale=2.5, rc={"legend.fontsize": 15})

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        x=[-2, 2],
        y=[1 / NUM_CLASSES, 1 / NUM_CLASSES],
        color=(0.1, 0.1, 0.1, 0.5),
        label="Baseline",
    )
    ax.lines[0].set_linestyle("--")

    sns.lineplot(
        x="Mutation Probability",
        y="Self-play Performance",
        data=self_play_df,
        label="Self-play Performance",
        color=sns.color_palette()[1],
    )

    sns.lineplot(
        x="Mutation Probability",
        y="Zero-Shot Coordination Score",
        data=zs_coord_df,
        label="Zero-shot Performance",
        palette=sns.color_palette()[0],
    )
    sns.scatterplot(
        x="Mutation Probability",
        y="Zero-Shot Coordination Score",
        data=zs_coord_df,
        marker="x",
        palette=sns.color_palette()[0],
    )

    # sns.scatterplot(x='Mutation Probability', y='Self-play Performance', data=self_play_df, marker='x')
    sns.scatterplot(
        x="x",
        y="y",
        data=pd.DataFrame([{"x": 100, "y": 100}]),
        color=(0.1, 0.1, 0.1, 0.5),
        marker="x",
        label="Raw Data",
    )
    plt.ylim([0, 1.05])
    plt.xlim([-0.05, 1.05])
    # plt.title('The Effect of Mutations on Zero-Shot Coordination')
    plt.ylabel("Performance")
    plt.xlabel("Mutation Probability")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(loc=4)


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/p_mutate_zsc").glob("data_*.csv")
    ]
    create_message_mutation_zsc_plot(*data)
    plt.savefig("figures/p_mutate_zsc/p_mutate_zsc.pdf", bbox_inches="tight", dpi=1000)


if __name__ == "__main__":
    reproduce_figure()
