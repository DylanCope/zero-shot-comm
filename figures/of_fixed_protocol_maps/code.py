from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot_protocol_maps(mean_class_message_map_df, mean_index_message_map_df):
    set_plotting_style(font_scale=2.0)
    _, axs = plt.subplots(
        1, 3, gridspec_kw={"width_ratios": [10, 10, 1], "wspace": 0.1}, sharex="col"
    )

    sns.heatmap(mean_class_message_map_df, vmin=0, vmax=1, ax=axs[0], cbar=False)
    axs[0].set_xlabel("Class")
    axs[0].set_ylabel("Symbol")

    sns.heatmap(mean_index_message_map_df, vmin=0, vmax=1, ax=axs[1], cbar_ax=axs[2])
    axs[1].set_xlabel("Time Step")
    axs[1].set_ylabel("")
    axs[1].set_yticks([])

    plt.tight_layout()


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/of_fixed_protocol_maps").glob("data_*.csv")
    ]
    plot_protocol_maps(*data)
    plt.savefig(
        "figures/of_fixed_protocol_maps/of_fixed_protocol_maps.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
