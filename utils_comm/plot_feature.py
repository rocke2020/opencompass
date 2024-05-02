import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from utils_comm.log_util import logger
import numpy as np
from pathlib import Path
from scipy import stats
import math
from typing import List, Union


amino_acids = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "L",
    "M",
    "N",
    "P",
    "K",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]


def kde_plot_feature(
    dfs,
    feature_name="HydrophRatio",
    vertical_line_in_x=[],
    save_path=None,
    labels=["pos", "neg"],
    manual_xticks=None,
    title_prefix="",
    title="Kernel Distribution Estimation",
    label_fontsize=12,
    title_fontsize=15,
    xticks_fontsize=11,
    legend_fontsize=11,
):
    """
    manual_xticks: example, [-0.35, 0, 0.35, 0.7, 1.05, 1.4]
    """
    plt.figure(figsize=(6.229, 4.463))
    sns.set_style(style="ticks")
    sns.set_context("talk")
    markersize = 4
    for i, df in enumerate(dfs):
        df[feature_name].plot.kde(
            label=labels[i],
            linewidth=1,
            # marker = "o", markersize=markersize, markevery=10,
        )
    for vertical_line_location_in_x in vertical_line_in_x:
        plt.axvline(
            x=vertical_line_location_in_x,
            c="orange",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )

    if manual_xticks:
        # keep the same format as plt.xticks()
        orig_xticks = [manual_xticks, 0]
    else:
        orig_xticks = plt.xticks()
    orig_locs = add_vertical_line_values_into_xticks(vertical_line_in_x, orig_xticks)
    plt.xticks(orig_locs, fontsize=xticks_fontsize)
    plt.yticks(fontsize=xticks_fontsize)

    plt.legend(loc="upper right", frameon=0, framealpha=0.5, fontsize=legend_fontsize)
    if title_prefix:
        title = title_prefix + " " + title.lower()
    plt.title(
        title, pad=10, loc="right", fontsize=title_fontsize, fontname="Times New Roman"
    )
    plt.xlabel(f"{feature_name} value", fontdict={"size": label_fontsize})
    plt.ylabel("Density", fontdict={"size": label_fontsize})
    sns.despine()
    plt.show()
    if save_path:
        logger.info(f"Save img file {save_path}")
        plt.savefig(save_path, bbox_inches="tight")


def kde_plot_feature_one_df(
    df,
    task_name,
    feature_name="len",
    vertical_line_in_x=[15],
    save_img_file=None,
    print_summary=False,
):
    plt.figure(figsize=(10, 6))
    sns.set_style(style="ticks")
    sns.set_context("talk")
    markersize = 4
    df[feature_name].plot.kde(
        label=f"{task_name} {feature_name}",
        c="darkcyan",
        linewidth=1,
        marker="o",
        markersize=markersize,
        markevery=10,
    )

    for vertical_line_location_in_x in vertical_line_in_x:
        plt.axvline(
            x=vertical_line_location_in_x,
            c="green",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )
    if vertical_line_in_x:
        orig_locs = add_vertical_line_values_into_xticks(
            vertical_line_in_x, plt.xticks()
        )
        plt.xticks(orig_locs)

    plt.legend(loc="upper right", frameon=0, framealpha=0.5, fontsize="x-small")
    sns.despine()
    plt.tight_layout(pad=2)
    plt.xlabel(f"{feature_name} value")
    plt.ylabel(f"{feature_name} kde density")
    plt.show()
    if save_img_file:
        plt.savefig(save_img_file)
    if print_summary:
        logger.info(df[feature_name].describe())


def add_vertical_line_values_into_xticks(vertical_line_in_x, orig_xticks):
    orig_locs = orig_xticks[0]
    new_locs = []
    for vertical_locs in vertical_line_in_x:
        for locs in orig_locs:
            if locs <= vertical_locs:
                new_locs.append(locs)
            else:
                new_locs.append(vertical_locs)
                new_locs.append(locs)
        orig_locs = new_locs
        new_locs = []
    return orig_locs


def plot_proba_hist_kde(
    inputs,
    title,
    save_img_file=None,
    vertical_line_in_x=None,
    x_locs=None,
    enable_kde=False,
    bins: Union[str, int] = "auto",
    x_label="",
    change_small_to_zero=False,
):
    """ """
    normalized_inputs = inputs
    if change_small_to_zero:
        # A bug in sns.histplot, we have to if item < 1e-5: item = 0. This bug disappears.
        normalized_inputs = []
        for item in inputs:
            if item < 1e-5:
                item = 0
            normalized_inputs.append(item)

    plt.figure(figsize=(12, 10))
    sns.set_style(style="ticks")
    sns.set_context("talk")
    logger.info(f"Starts to plot hist, with kde {enable_kde}")
    sns.histplot(data=normalized_inputs, kde=enable_kde, bins=bins, discrete=True)
    if vertical_line_in_x:
        for vertical_line_location_in_x in vertical_line_in_x:
            plt.axvline(
                x=vertical_line_location_in_x,
                c="green",
                linestyle="--",
                linewidth=1,
                alpha=0.5,
            )
    if x_locs:
        plt.xticks(x_locs)
    x_ticks_font = 20
    plt.xticks(fontsize=x_ticks_font)
    plt.title(title, fontsize=28)
    plt.legend(loc="upper right", frameon=0, framealpha=0.5, fontsize="x-small")
    sns.despine()
    plt.tight_layout(pad=2)
    if not x_label:
        x_label = "value"
    label_font = x_ticks_font + 2
    plt.xlabel(x_label, fontsize=label_font)
    plt.ylabel("count", fontsize=label_font)
    if save_img_file:
        plt.savefig(save_img_file)
    logger.info("Plots hist ends")


def hist_2_df_comparison(
    pos_df,
    neg_df,
    task_name,
    feature_name="",
    save_path=None,
    histtype="step",
    print_summary=False,
    bins=25,
):
    plt.figure(figsize=(16, 10))
    sns.set_style(style="ticks")
    sns.set_context("talk")
    if feature_name == "":
        feature_name = task_name
    neg_df[feature_name].hist(
        label=f"neg {task_name} {feature_name}", histtype=histtype, bins=bins
    )
    pos_df[feature_name].hist(
        label=f"pos {task_name} {feature_name}", histtype=histtype, bins=bins
    )
    plt.legend(loc="upper right", frameon=0, framealpha=0.5, fontsize="x-small")
    sns.despine()
    plt.tight_layout(pad=2)
    plt.ylabel("count")
    plt.xlabel("value")
    # plt.xticks([0, 0.38, 0.5, 1])
    plt.show()
    if save_path:
        plt.savefig(save_path)
    if print_summary:
        logger.info(f"neg_df {feature_name} summary describe()")
        logger.info(neg_df[feature_name].describe())
        logger.info(f"\npos_df {feature_name} summary describe()")
        logger.info(pos_df[feature_name].describe())


def hist_plot_feature_general(
    dfs,
    label_prefixes,
    feature,
    title,
    save_path=None,
    histtype="step",
    bins=15,
    print_summary=False,
    fontsize="medium",
):
    """bins should not be larger than the unqiue item num, otherwise there will be abnormal gap in plots."""
    assert len(dfs) == len(label_prefixes)
    plt.figure(figsize=(16, 10))
    sns.set_style(style="ticks")
    sns.set_context("talk")
    for df, label_prefix in zip(dfs, label_prefixes):
        df[feature].hist(
            label=f"{label_prefix} {feature}", histtype=histtype, bins=bins
        )
    plt.legend(loc="upper right", frameon=0, framealpha=0.5, fontsize=fontsize)
    sns.despine()
    plt.tight_layout(pad=2)
    plt.ylabel("count")
    plt.xlabel("value")
    plt.title(title, pad=0, loc="center", fontsize=18, fontname="Times New Roman")
    # plt.xticks([0, 0.38, 0.5, 1])
    plt.show()
    if save_path:
        plt.savefig(save_path)
    if print_summary:
        for df, label_prefix in zip(dfs, label_prefixes):
            logger.info(f"{label_prefix} {feature} summary describe()")
            logger.info(df[feature].describe())


def calc_count_with_all_key(df, column, all_keys_in_fixed_order):
    """NB: all_keys_in_fixed_order, we need pre-run to initialize the dict to keep fixed sequence"""
    count = {}
    for v in all_keys_in_fixed_order:
        count[v] = 0
    for value in df[column]:
        count[round(value)] += 1
    return count


def plot_bars(
    values, xlabels, ylabel, title, save_img_file, x_fontzie=None, figsize=(13, 9)
):
    """ """
    plt.figure(figsize=figsize)
    sns.set_style(style="ticks")
    sns.set_context("talk")
    x = np.arange(len(xlabels))
    width = 0.35
    plt.bar(x - width / 2, values, width, label="values")
    plt.ylabel(ylabel)
    plt.title(title, pad=10)
    if x_fontzie:
        plt.xticks(x, xlabels, fontsize=x_fontzie)
    else:
        plt.xticks(x, xlabels)
    plt.legend()
    ## there is abnormal display error when the value is < 1, skip to add text
    # for x1, y1 in enumerate(pos_values):
    #     plt.text(x1 - width/2, y1+1, round(y1, 1), fontsize=8, ha='center')
    # for x2, y2 in enumerate(neg_values):
    #     plt.text(x2 + width/2, y2+1, round(y1, 2), fontsize=8, ha='center')
    plt.show()
    if save_img_file:
        plt.savefig(save_img_file)


def calc_spearmanr(x, y, notes=""):
    """ """
    res = stats.spearmanr(x, y)
    logger.info(f"{notes} spearmanr: {res}")
    if hasattr(res, "correlation"):
        spearman_ratio = float(res.correlation)  # type: ignore
    else:
        assert hasattr(res, "statistic")
        spearman_ratio = float(res.statistic)  # type: ignore
    if hasattr(res, "pvalue"):
        pvalue = float(res.pvalue)  # type: ignore
        logger.info(f"{notes} spearmanr pvalue: {pvalue}")
    logger.info(f"{notes} spearman_ratio: {spearman_ratio}")
    return spearman_ratio


def plot_scatter(
    x,
    y,
    save_file,
    title="",
    fontsize=16,
    plot_diagonal=True,
    value_name="value",
    xlabel="",
    ylabel="",
    calc_spearman=True,
):
    """
    plot performance of regression model against validation dataset, x can be experimental and y predicted values.
    title with f'spearman correlation ratio {spearmanr}'
    """
    plt.figure(figsize=(16, 10))
    plt.scatter(x, y)
    plt.grid(visible=True)
    if not xlabel:
        xlabel = f"Real {value_name}"
    if not ylabel:
        ylabel = f"Predicted {value_name}"
    plt.xlabel(xlabel, fontdict={"size": fontsize})
    plt.ylabel(ylabel, fontdict={"size": fontsize})
    spearman_ratio = None
    if calc_spearman:
        spearman_ratio = calc_spearmanr(x, y)
        if not title:
            title = f"spearman correlation ratio {spearman_ratio:.4f}"
        else:
            title = title + f", spearman correlation ratio {spearman_ratio:.4f}"
        logger.info(title)
    if title:
        plt.title(title, fontdict={"size": fontsize + 2})
    if plot_diagonal:
        x = plt.xticks()[0]
        plt.plot(x, x, c="green", linestyle="--", linewidth=1, alpha=0.5)
    plt.tick_params(labelsize=14)
    plt.savefig(save_file)
    return spearman_ratio


def plot_lines(ranking_scores, recalls, f1s, precisions, plot_dir):
    """An example to plot multi plines"""
    title = "recall_precision_with_different_ranking scores"
    fontsize = 16
    plt.figure(figsize=(16, 10))
    plt.plot(ranking_scores, recalls, c="blue", label="recall")
    plt.plot(ranking_scores, f1s, c="cyan", label="f1")
    plt.plot(ranking_scores, precisions, c="green", label="precision")
    plt.grid(visible=True)
    plt.title(title, fontdict={"size": fontsize + 2})
    plt.legend(fontsize=fontsize)
    plt.xlabel("ranking scores", fontsize=fontsize)
    plt.ylabel("recall precision ratio", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(plot_dir / f"{title}.png")


def plot_violin(
    x_column,
    y_column,
    data,
    img_file,
    title_fontsize=15,
    label_fontsize=12,
    xticks_fontsize=11,
):
    """ """
    init_plot_style()
    sns.violinplot(x=x_column, y=y_column, data=data, inner="quartile")
    plt.title(y_column, loc="center", fontsize=title_fontsize)
    sns.despine()
    plt.xticks(fontsize=label_fontsize)
    plt.yticks(fontsize=xticks_fontsize)

    plt.xlabel("")
    plt.ylabel(y_column, fontdict={"size": label_fontsize})
    plt.savefig(img_file, bbox_inches="tight")


def init_plot_style(style="ticks", context="talk"):
    """ """
    sns.set_style(style=style)
    sns.set_context(context)


if __name__ == "__main__":
    pass
