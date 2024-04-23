from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


X_NAME = "Mutation Probability"


def plot_metrics_with_p_mutate(zs_coord_df, responsiveness_df):
    # set_plotting_style()
    set_plotting_style(font_scale=2.5, rc={"legend.fontsize": 18})

    # plt.figure(figsize=(8, 4))
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        x=X_NAME,
        y="Zero-Shot Coordination Score",
        data=zs_coord_df,
        label="Zero-shot Performance",
    )
    sns.lineplot(
        x=X_NAME,
        y="Teacher Responsiveness",
        data=responsiveness_df,
        label="Teacher Responsiveness",
    )
    sns.lineplot(
        x=X_NAME,
        y="Student Responsiveness",
        data=responsiveness_df,
        label="Student Responsiveness",
    )
    sns.lineplot(
        x=X_NAME, y="Protocol Diversity", data=zs_coord_df, label="Protocol Diversity"
    )

    sns.scatterplot(
        x=X_NAME,
        y="Zero-Shot Coordination Score",
        data=zs_coord_df.groupby(X_NAME).mean(),
        # label="Zero-shot Performance",
    )
    sns.scatterplot(
        x=X_NAME,
        y="Teacher Responsiveness",
        data=responsiveness_df.groupby(X_NAME).mean(),
        # label="Teacher Responsiveness",
    )
    sns.scatterplot(
        x=X_NAME,
        y="Student Responsiveness",
        data=responsiveness_df.groupby(X_NAME).mean(),
        # label="Student Responsiveness",
    )
    sns.scatterplot(
        x=X_NAME,
        y="Protocol Diversity",
        data=zs_coord_df.groupby(X_NAME).mean(),
        # label="Protocol Diversity"
    )

    plt.ylim([-0.05, 1.05])

    # plt.xlim([1.95, CHANNEL_SIZE + .05])
    # plt.title('The Effect of Channel Permutation on Zero-Shot Coordination')
    plt.ylabel("")
    plt.xlabel("Mutation Probability")
    # plt.xticks(pd.unique(zs_coord_df[X_NAME]))


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/msg_mutation_experiments_metrics").glob(
            "data_*.csv"
        )
    ]
    plot_metrics_with_p_mutate(*data)
    plt.savefig(
        "figures/msg_mutation_experiments_metrics/msg_mutation_experiments_metrics.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
