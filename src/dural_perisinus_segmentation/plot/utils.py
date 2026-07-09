from __future__ import annotations

from enum import Enum
from itertools import combinations
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.stats import bootstrap, pearsonr, ttest_ind, ttest_rel

from ..stat.conf import compute_t_conf
from ..stat.pvalues import pval_to_stars

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

plt.style.use("ggplot")
PALETTE = sns.color_palette("colorblind")


def bland_altman_plots(
    df: pd.DataFrame,
    plots: Sequence[tuple[str, str]],
    quantity: str,
    unit: str,
    hue: Optional[str] = None,
    hue_order: Optional[Sequence[str]] = None,
    stats_x_pos: Optional[Sequence[float]] = None,
    titles: Optional[Sequence[str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    grid_spacing: Optional[float] = None,
    *args,
    **kwargs,
) -> tuple[Figure, Axes]:
    """
    Multiple Bland-Altman plots (https://en.wikipedia.org/wiki/Bland%E2%80%93Altman_plot).

    Parameters
    ----------
    df : pd.DataFrame
        The data.
    plots : Sequence[tuple[str, str]]
        The two distributions to compare for each plot.
    quantity : str
        A name for the quantity.
    unit : str
        A unit for the quantity.
    hue : Optional[str], default=None
        To plot points of different color depending on a category.
    hue_order : Optional[Sequence[str]], default=None
        The order the categories.
    stats_x_pos : Optional[float], default=None
        Position where the mean value and limits of agreements are printed for each plot.
        If ``None``, they are not printed.
    titles : Optional[Sequence[str]], default=None
        A title for each plot.
    figsize : Optional[tuple[float, float]], default=None
        The size of the figure.
    grid_spacing : Optional[float], default=None
        THe spacing for the x-axis and y-axis grid lines.

    Returns
    -------
    tuple[Figure, Axes]
    """
    f, ax = plt.subplots(1, len(plots), figsize=figsize, sharey=True)
    for i, ((x, y), title, pos) in enumerate(zip(plots, titles, stats_x_pos)):
        _bland_altman_plot(
            df,
            x,
            y,
            quantity=quantity,
            unit=unit,
            ax=ax[i],
            hue=hue,
            hue_order=hue_order,
            stats_x_pos=pos,
            legend=(i == 0),
            y_label=(i == 0),
            title=title,
            grid_spacing=grid_spacing,
        )

    return f, ax


def _bland_altman_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    quantity: str,
    unit: str,
    ax: Optional[Axes] = None,
    hue: Optional[str] = None,
    hue_order: Optional[Sequence[str]] = None,
    stats_x_pos: Optional[float] = None,
    legend: bool = True,
    y_label: bool = True,
    title: str = None,
    grid_spacing: Optional[float] = None,
    *args,
    **kwargs,
) -> None:
    """
    Bland-Altman plot with confidence interval for the mean and limits of agreements
    (CIs computed with bootstrap).
    """
    average = np.mean([df[x], df[y]], axis=0)
    diff = df[y] - df[x]
    dict_ = {"diff": diff, "avg": average}

    if hue is not None:
        dict_[hue] = df[hue]

    df = pd.DataFrame(dict_)

    bs = bootstrap(
        data=(diff.dropna(),),
        statistic=lambda x: (
            np.mean(x),
            np.mean(x) - 1.96 * np.std(x),
            np.mean(x) + 1.96 * np.std(x),
        ),
        paired=True,
        vectorized=False,
        n_resamples=9999,
        method="percentile",
        confidence_level=0.95,
    )

    mean = bs.bootstrap_distribution[0].mean()
    mean_ci = (
        bs.confidence_interval.low[0],
        bs.confidence_interval.high[0],
    )
    loa_low = bs.bootstrap_distribution[1].mean()
    loa_low_ci = (
        bs.confidence_interval.low[1],
        bs.confidence_interval.high[1],
    )
    loa_high = bs.bootstrap_distribution[2].mean()
    loa_high_ci = (
        bs.confidence_interval.low[2],
        bs.confidence_interval.high[2],
    )

    ax = sns.scatterplot(
        df,
        x="avg",
        y="diff",
        hue=hue,
        hue_order=hue_order,
        ax=ax,
        palette=PALETTE,
        *args,
        **kwargs,
    )
    ax.axhline(mean, color=PALETTE[2], linestyle="--")
    ax.axhspan(mean_ci[0], mean_ci[1], color=PALETTE[2], alpha=0.1)
    ax.axhline(loa_low, color=PALETTE[3], linestyle="--")
    ax.axhspan(loa_low_ci[0], loa_low_ci[1], color=PALETTE[3], alpha=0.1)
    ax.axhline(loa_high, color=PALETTE[3], linestyle="--")
    ax.axhspan(loa_high_ci[0], loa_high_ci[1], color=PALETTE[3], alpha=0.1)
    ax.set_xlabel(f"Average {quantity} ({unit})")
    ax.set_ylabel(f"{quantity.capitalize()} difference ({unit})")
    leg = ax.legend(title=None)

    if title:
        ax.set_title(title, fontsize=10)

    if stats_x_pos is not None:
        ax.text(
            x=stats_x_pos, y=mean, s=f"mean: {mean:.2f}", color=PALETTE[2], va="center"
        )
        ax.text(
            x=stats_x_pos,
            y=loa_low,
            s=f"-1.96 SD: {loa_low:.2f}",
            color=PALETTE[3],
            va="center",  # vertically align with the line
        )
        ax.text(
            x=stats_x_pos,
            y=loa_high,
            s=f"+1.96 SD: {loa_high:.2f}",
            color=PALETTE[3],
            va="center",
        )

    if not legend:
        leg.remove()

    if not y_label:
        ax.set_ylabel("")

    if grid_spacing is not None:
        ax.yaxis.set_major_locator(MultipleLocator(grid_spacing))
        ax.yaxis.set_major_locator(MultipleLocator(grid_spacing))


def scatterplots(
    df: pd.DataFrame,
    plots: Sequence[tuple[str, str]],
    hue: Optional[str] = None,
    hue_order: Optional[Sequence[str]] = None,
    min_: Optional[float] = None,
    max_: Optional[float] = None,
    print_metrics: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    grid_spacing: Optional[float] = None,
) -> tuple[Figure, Axes]:
    f, ax = plt.subplots(1, len(plots), figsize=figsize)
    for i, (x, y) in enumerate(plots):
        _scatterplot(
            df,
            x,
            y,
            hue=hue,
            hue_order=hue_order,
            min_=min_,
            max_=max_,
            print_metrics=print_metrics,
            ax=ax[i],
            grid_spacing=grid_spacing,
            legend=(i == 0),
        )

    return f, ax


def _scatterplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    hue_order: Optional[Sequence[str]] = None,
    min_: Optional[float] = None,
    max_: Optional[float] = None,
    print_metrics: bool = True,
    ax: Optional[Axes] = None,
    grid_spacing: Optional[float] = None,
    legend: bool = True,
) -> Axes:
    """
    A scatter plot with metrics.
    """
    if min_ is None:
        min_ = min(df[x].min(), df[y].min())
    if max_ is None:
        max_ = max(df[x].max(), df[y].max())

    ax = sns.scatterplot(
        df,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        palette=PALETTE,
        ax=ax,
    )
    seaborn_legend = ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.plot(
        np.linspace(min_, max_), np.linspace(min_, max_), c=PALETTE[2], linestyle="--"
    )

    if grid_spacing is not None:
        ax.xaxis.set_major_locator(MultipleLocator(grid_spacing))
        ax.yaxis.set_major_locator(MultipleLocator(grid_spacing))

    if print_metrics:
        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color="none",
                    label=r"$\bf{{VER}}$: {}".format(_get_ver(df, x, y)[0]),
                ),
                Line2D(
                    [],
                    [],
                    color="none",
                    label=r"$\bf{{AVER}}$: {}".format(_get_aver(df, x, y)[0]),
                ),
                Line2D(
                    [],
                    [],
                    color="none",
                    label=r"$\bf{{Pearson \ r}}$: {}".format(
                        _get_pearson_str(df, x, y)
                    ),
                ),
            ],
            loc="lower right",
            handlelength=0,
        )
    ax.add_artist(seaborn_legend)

    if not legend:
        seaborn_legend.remove()

    return ax


def get_pvalues(
    df: pd.DataFrame,
    comparisons: Sequence[tuple[tuple[str, str], tuple[str, str]]],
    quantity: str,
    mode: Optional[str | TtestMode] = "independent",
) -> None:
    """
    Compute p-values of t-test comparing AVER and VER distributions.

    Compute also p-values of the individual distributions.

    Parameters
    ----------
    df : pd.DataFrame
        The data.
    comparisons : Sequence[tuple[tuple[str, str], tuple[str, str]]]
        Name of the columns to compare. If [((a, b), (c, d)), ...] is passed,
        VER(a, b) and VER(c, d) will be computed and the resulting distributions
        compared via a t-test. The individual distributions a, b, c, and d will also be compared
        with one another.
    quantity : str
        A name for the quantity (e.g. 'volume').
    mode : Optional[str  |  TtestMode], default=independent
        Type of t-test. "related" or "independent"
    """
    mode = TtestMode(mode)
    print("P-values:")
    for pair1, pair2 in comparisons:
        print("-" * 5)
        print(
            f"VER({pair1}) vs VER({pair2}): {mode.get_test()(_get_ver(df, pair1[0], pair1[1])[1], _get_ver(df, pair2[0], pair2[1])[1])[1]}"
        )
        print(
            f"AVER({pair1}) vs AVER({pair2}): {mode.get_test()(_get_aver(df, pair1[0], pair1[1])[1], _get_aver(df, pair2[0], pair2[1])[1])[1]}"
        )

    print("=" * 5)
    individual_elem = set([x for z in comparisons for y in z for x in y])
    for x, y in combinations(individual_elem, 2):
        print(f"{quantity}({x}) vs {quantity}({y}): {mode.get_test()(df[x], df[y])[1]}")


def _get_ver(df: pd.DataFrame, x: str, y: str) -> tuple[str, np.ndarray]:
    """
    Compute VER and format the result.
    """
    ver = 2 * (df[y] - df[x]) / (df[x] + df[y])
    conf_ver = compute_t_conf(ver)
    return f"{ver.mean():.2f} [{conf_ver[0]:.2f}, {conf_ver[1]:.2f}]", ver.to_numpy()


def _get_aver(df: pd.DataFrame, x: str, y: str) -> tuple[str, np.ndarray]:
    """
    Compute AVER and format the result.
    """
    aver = 2 * (df[y] - df[x]).abs() / (df[x] + df[y])
    conf_aver = compute_t_conf(aver)
    return (
        f"{aver.mean():.2f} [{conf_aver[0]:.2f}, {conf_aver[1]:.2f}]",
        aver.to_numpy(),
    )


def _get_pearson_str(df: pd.DataFrame, x: str, y: str) -> str:
    """
    Compute pearson correlation and format the result.
    """
    df = df.dropna().sort_values("participant_id")
    pearson = pearsonr(df[x], df[y])
    return f"{pearson.statistic:.2f} [{pearson.confidence_interval().low:.2f}, {pearson.confidence_interval().high:.2f}]"


def boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    order: Optional[Sequence[str]] = None,
    hue_order: Optional[Sequence[str]] = None,
    plot_points: bool = True,
    plot_tests: bool = True,
    test_mode: str | TtestMode = "independent",
    y_space_above_last_value: float = 0.01,
    pairs_hue: Optional[Sequence[tuple[int, int]]] = None,
    figsize: Optional[tuple[float, float]] = None,
    legend_loc: Optional[str] = None,
    bbox_to_anchor: tuple[float, float] = None,
) -> tuple[Figure, Axes]:
    """
    Boxplot with seaborn.

    Parameters
    ----------
    df : pd.DataFrame
        The data.
    x : str
        Variable in ``df`` for the x axis.
    y : str
        Variable in ``df`` for the y axis.
    hue : str
        Variable in ``df`` for plotting distributions of different categories.
    order : Optional[Sequence[str]], default=None
        Order for the x axis.
    hue_order : Optional[Sequence[str]], default=None
        Order for the distributions.
    plot_points : bool, default=True
        Whether to plot the points of the distributions.
    plot_tests : bool, default=True
        Whether to perform t-tests and plot the p-values significance.
    test_mode : str | TtestMode, default="independent"
        Type of test to perform. Either "related" or "independent".
    y_space_above_last_value : float, default=0
        The space between the last y value and the comparison bar.
    pairs_hue : Optional[Sequence[tuple[int, int]]], default=None
        Pairs on which the t-tests must be performed.
    figsize : Optional[tuple[float, float]], default=None
        The size of the figure.
    legend_loc : Optional[str], default=None
        The localisation of the legend.
    bbox_to_anchor : tuple[float, float], default=None
        To precisely set the localisation of the legend.

    Returns
    -------
    tuple[Figure, Axes]
    """
    f, ax = _boxplot(
        df,
        x,
        y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        figsize=figsize,
        legend_loc=legend_loc,
        bbox_to_anchor=bbox_to_anchor,
    )
    if plot_points:
        _swarmplot(
            ax,
            df,
            x,
            y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            legend_loc=legend_loc,
            bbox_to_anchor=bbox_to_anchor,
        )
    if plot_tests:
        _plot_p_values(
            ax,
            df,
            y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            mode=test_mode,
            pairs_hue=pairs_hue,
            y_space_above_last_value=y_space_above_last_value,
        )

    return f, ax


def _boxplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    order: Optional[Sequence[str]] = None,
    hue_order: Optional[Sequence[str]] = None,
    figsize: Optional[tuple[float, float]] = None,
    legend_loc: Optional[str] = None,
    bbox_to_anchor: tuple[float, float] = None,
) -> tuple[Figure, Axes]:
    """
    A boxplot with seaborn.

    'order' is the order of the values on the x-axis.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        ax=ax,
        showmeans=True,
        boxprops={"alpha": 0.4},
        palette=PALETTE,
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": "5",
        },
    )
    ax.legend(loc=legend_loc, title=None, bbox_to_anchor=bbox_to_anchor)

    return fig, ax


def _swarmplot(
    ax: Axes,
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    order: Optional[Sequence[str]] = None,
    hue_order: Optional[Sequence[str]] = None,
    legend_loc: Optional[str] = None,
    bbox_to_anchor: tuple[float, float] = None,
):
    """
    Swarmplot with seaborn to add points to a boxplot.

    'bbox_to_anchor' enables to control the exact localisation of the legend.
    """
    sns.swarmplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        ax=ax,
        alpha=0.8,
        palette=PALETTE,
        dodge=True,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=[
            (handles[i], handles[j])
            for i, j in zip(
                range(0, len(hue_order)), range(len(hue_order), 2 * len(hue_order))
            )
        ],
        labels=labels,
        loc=legend_loc,
        handlelength=4,
        bbox_to_anchor=bbox_to_anchor,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )


class TtestMode(str, Enum):
    """
    Kind of t-test.
    """

    REL = "related"
    INDEP = "independent"

    def get_test(self) -> Callable:
        if self.value == TtestMode.INDEP:
            test = ttest_ind
        else:
            test = ttest_rel

        return lambda a, b: test(a, b, nan_policy="omit")


def _plot_p_values(
    ax: Axes,
    df: pd.DataFrame,
    y: str,
    hue: str,
    order: Sequence[str],
    hue_order: Sequence[str],
    mode: TtestMode,
    pairs_hue: Optional[Sequence[tuple[int, int]]] = None,
    y_space_above_last_value: float = 0,
) -> None:
    """
    Add p-values significance on a boxplot.

    mode can be either "related" t-test on related samples, or "independent" for independent
    samples.
    pairs_hue defines on which pairs the t-test should be performed.
    y_space_above_last_value defines the space between the last value and the comparison bar.
    """
    mode = TtestMode(mode)

    if not pairs_hue:
        pairs_hue = list(combinations(range(len(hue_order)), 2))

    bar_h = 0.01  # size of the bar
    y_space_between_bars = 0.03  # steps between bars
    y_space_above_bar = 0.01

    y_top = df[y].max() + y_space_above_last_value
    print("P-values")
    for i, metric in enumerate(order):
        sub = df[df["metric"] == metric].sort_values("participant_id")

        for k, (hi, hj) in enumerate(pairs_hue):
            vals1 = sub[sub[hue] == hue_order[hi]][y].values
            vals2 = sub[sub[hue] == hue_order[hj]][y].values

            _, p = mode.get_test()(vals1, vals2)
            print(f"{hue_order[hi]} vs {hue_order[hj]}: ", p)

            _draw_bar(
                ax,
                x1=_box_x(i, hi, len(hue_order)),
                x2=_box_x(i, hj, len(hue_order)),
                y=y_top + k * y_space_between_bars,
                label=pval_to_stars(p),
                bar_h=bar_h,
            )
    ax.set_ylim(
        top=y_top + len(pairs_hue) * y_space_between_bars + bar_h + y_space_above_bar
    )


def _box_x(metric_idx: int, hue_idx: int, n_hues: int) -> float:
    """
    Compute the x-axis position of a box given its metric and hue indices.
    """
    total_width = 0.8  # seaborn default
    box_width = total_width / n_hues
    offset = -total_width / 2 + box_width / 2 + hue_idx * box_width

    return metric_idx + offset


def _draw_bar(
    ax: Axes,
    x1: float,
    x2: float,
    y: float,
    label: str,
    bar_h: float = 0.01,
    fontsize: float = 11,
) -> None:
    """
    Draw a significance bar between x1 and x2 at height y, whose height is bar_h.
    Add a label above the bar.
    """
    ax.plot([x1, x1, x2, x2], [y, y + bar_h, y + bar_h, y], lw=1.5, color="k")
    ax.text(
        (x1 + x2) / 2, y + bar_h, label, ha="center", va="bottom", fontsize=fontsize
    )
