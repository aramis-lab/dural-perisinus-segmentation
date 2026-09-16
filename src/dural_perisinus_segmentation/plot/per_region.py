# To plot the results per region. Figure S4.

# %%
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dural_perisinus_segmentation.plot.utils import (
    _get_pearson_str,
    boxplot,
    get_aver,
    get_mean_and_conf,
    get_ver,
    plot_table,
)

BIDS_OUT = Path(
    "../../../data/nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/bids_pred/"
)
BIDS_IN = Path("../../../data/bids")

RATER1_KEY = "SL"
RATER2_KEY = "DD"

EVALUATION_RATER_1 = BIDS_OUT / "desc-SL_label-lymph_evaluationDetails.tsv"
EVALUATION_RATER_2 = BIDS_OUT / "desc-DD_label-lymph_evaluationDetails.tsv"
EVALUATION_RATER_1_ROI = (
    BIDS_OUT / f"desc-{RATER1_KEY}_label-lymphRoi_evaluationRoiDetails.tsv"
)
EVALUATION_RATER_2_ROI = (
    BIDS_OUT / f"desc-{RATER2_KEY}_label-lymphRoi_evaluationRoiDetails.tsv"
)
INTER_RATER_COMPARISON = (
    BIDS_IN / f"rater1-{RATER1_KEY}_rater2-{RATER2_KEY}_interraterDetails.tsv"
)
INTER_RATER_COMPARISON_ROI = (
    BIDS_IN / f"rater1-{RATER1_KEY}_rater2-{RATER2_KEY}_interraterRoiDetails.tsv"
)

PRED_VOLUMES = BIDS_OUT / "volumesDetails.tsv"
RATER1_VOLUMES = BIDS_IN / f"rater-{RATER1_KEY}_volumesDetails.tsv"
RATER2_VOLUMES = BIDS_IN / f"rater-{RATER2_KEY}_volumesDetails.tsv"
RATER1_VOLUMES_ROI = BIDS_IN / f"rater-{RATER1_KEY}_volumesRoiDetails.tsv"
RATER2_VOLUMES_ROI = BIDS_IN / f"rater-{RATER2_KEY}_volumesRoiDetails.tsv"
PRED_VOLUMES_ROI_RATER1 = (
    BIDS_OUT / f"desc-{RATER1_KEY}_label-lymphRoi_volumesRoiDetails.tsv"
)
PRED_VOLUMES_ROI_RATER2 = (
    BIDS_OUT / f"desc-{RATER2_KEY}_label-lymphRoi_volumesRoiDetails.tsv"
)

# %%
model_evaluation_rater1 = pd.concat(
    [
        pd.read_csv(
            EVALUATION_RATER_1,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
        pd.read_csv(
            EVALUATION_RATER_1_ROI,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
    ],
    axis=1,
)
model_evaluation_rater1 = model_evaluation_rater1.rename(
    columns={k: RATER1_KEY + "_" + k for k in model_evaluation_rater1.columns}
)
model_evaluation_rater2 = pd.concat(
    [
        pd.read_csv(
            EVALUATION_RATER_2,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
        pd.read_csv(
            EVALUATION_RATER_2_ROI,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
    ],
    axis=1,
)
model_evaluation_rater2 = model_evaluation_rater2.rename(
    columns={k: RATER2_KEY + "_" + k for k in model_evaluation_rater2.columns}
)
inter_rater = pd.concat(
    [
        pd.read_csv(
            INTER_RATER_COMPARISON,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
        pd.read_csv(
            INTER_RATER_COMPARISON_ROI,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
    ],
    axis=1,
)
inter_rater = inter_rater.rename(
    columns={k: "inter-rater" + "_" + k for k in inter_rater.columns}
)
volumes_rater_1 = pd.concat(
    [
        pd.read_csv(
            RATER1_VOLUMES,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
        pd.read_csv(
            RATER1_VOLUMES_ROI,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
    ],
    axis=1,
)
volumes_rater_1 = volumes_rater_1.rename(columns={"total_volume": "volume"}).rename(
    columns={k: "volume" + "_" + k for k in volumes_rater_1.columns if k != "volume"}
)
volumes_rater_1 = volumes_rater_1.rename(
    columns={k: RATER1_KEY + "_" + k for k in volumes_rater_1.columns}
)
volumes_rater_2 = pd.concat(
    [
        pd.read_csv(
            RATER2_VOLUMES,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
        pd.read_csv(
            RATER2_VOLUMES_ROI,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
    ],
    axis=1,
)
volumes_rater_2 = volumes_rater_2.rename(columns={"total_volume": "volume"}).rename(
    columns={k: "volume" + "_" + k for k in volumes_rater_2.columns if k != "volume"}
)
volumes_rater_2 = volumes_rater_2.rename(
    columns={k: RATER2_KEY + "_" + k for k in volumes_rater_2.columns}
)
pred_volumes = (
    pd.read_csv(
        PRED_VOLUMES,
        sep="\t",
    )
    .set_index("participant_id")
    .drop(columns=["session_id"])
).rename(columns={"total_volume": "model-volume"})
model_volume_vs_rater1 = (
    pd.read_csv(
        PRED_VOLUMES_ROI_RATER1,
        sep="\t",
    )
    .set_index("participant_id")
    .drop(columns=["session_id"])
)
model_volume_vs_rater1 = model_volume_vs_rater1.rename(
    columns={
        k: RATER1_KEY + "_model-volume_" + k for k in model_volume_vs_rater1.columns
    }
)
model_volume_vs_rater2 = (
    pd.read_csv(
        PRED_VOLUMES_ROI_RATER2,
        sep="\t",
    )
    .set_index("participant_id")
    .drop(columns=["session_id"])
)
model_volume_vs_rater2 = model_volume_vs_rater2.rename(
    columns={
        k: RATER2_KEY + "_model-volume_" + k for k in model_volume_vs_rater2.columns
    }
)
df = pd.concat(
    [
        model_evaluation_rater1,
        model_evaluation_rater2,
        inter_rater,
    ],
    axis=1,
)
volumes = pd.concat(
    [
        volumes_rater_1,
        volumes_rater_2,
        pred_volumes,
        model_volume_vs_rater1,
        model_volume_vs_rater2,
    ],
    axis=1,
)
# %%
correlations = defaultdict(dict)

for suffix in ["", "_SSS", "_Torcular", "_StS", "_TS", "_SS"]:
    for rater in [RATER1_KEY, RATER2_KEY]:
        predicted_volume_key = (
            f"{rater}_model-volume" + suffix if suffix else "model-volume"
        )

        df[f"{rater}_aver" + suffix] = get_aver(
            volumes, f"{rater}_volume" + suffix, predicted_volume_key
        )[1]
        df[f"{rater}_ver" + suffix] = get_ver(
            volumes, f"{rater}_volume" + suffix, predicted_volume_key
        )[1]

        correlations[suffix][rater] = _get_pearson_str(
            volumes, f"{rater}_volume" + suffix, predicted_volume_key
        )

    df["inter-rater_aver" + suffix] = get_aver(
        volumes, f"{RATER1_KEY}_volume" + suffix, f"{RATER2_KEY}_volume" + suffix
    )[1]
    df["inter-rater_ver" + suffix] = get_ver(
        volumes, f"{RATER1_KEY}_volume" + suffix, f"{RATER2_KEY}_volume" + suffix
    )[1]

    correlations[suffix]["inter-rater"] = _get_pearson_str(
        volumes, f"{RATER1_KEY}_volume" + suffix, f"{RATER2_KEY}_volume" + suffix
    )

# %%
df = pd.melt(
    df.reset_index(), id_vars="participant_id", var_name="category", value_name="x"
)
df[["rater", "metric", "region"]] = df["category"].str.split("_", expand=True)
df = df.drop(columns=["category"])
df["region"] = df["region"].fillna("global")
df["rater"] = df["rater"].apply(
    lambda x: x.replace("DD", "test on DD").replace("SL", "test on SLn")
)

corr_df = pd.DataFrame(correlations)
corr_df.index = corr_df.index.map(
    lambda x: x.replace("DD", "test on DD").replace("SL", "test on SLn")
)
corr_df.columns = corr_df.columns.map(lambda x: x.replace("_", "") if x else "global")

# %%
hue_order = ["global", "SSS", "Torcular", "StS", "TS", "SS"]
metrics = ["dice", "cldice", "ver", "aver"]
metric_names = ["DSC ↑ ∈ [0, 1]", "clDice ↑ ∈ [0, 1]", "VER", "AVER ↓"]
rater_order = ["test on SLn", "test on DD", "inter-rater"]

f, ax = plt.subplots(2, 2, figsize=(15, 10))
for i, (metric, displayed_name) in enumerate(
    zip(
        metrics,
        metric_names,
    )
):
    boxplot(
        df[df["metric"] == metric],
        x="rater",
        y="x",
        hue="region",
        hue_order=hue_order,
        order=rater_order,
        plot_tests=False,
        figsize=(10, 5),
        plot_points=False,
        ax=ax[i // 2, i % 2],
    )
    ax[i // 2, i % 2].set_ylabel(displayed_name)
    ax[i // 2, i % 2].set_xlabel("")
    ax[i // 2, i % 2].legend(title="region")
    if i > 0:
        ax[i // 2, i % 2].get_legend().remove()

# %%
fig, axes = plt.subplots(5, 1, figsize=(15, 10))

for ax, metric, displayed_name in zip(
    axes[:-1],
    metrics,
    metric_names,
):
    df_metric = (
        df[df["metric"] == metric]
        .drop(columns=["metric"])
        .dropna()
        .pivot_table(
            columns="region",
            index="rater",
            values="x",
            aggfunc=lambda x: get_mean_and_conf(x, conf_format=".2f"),
        )
        .sort_index(axis=1, key=lambda x: x.map(lambda y: hue_order.index(y)))
        .sort_index(axis=0, key=lambda x: x.map(lambda y: rater_order.index(y)))
    )

    plot_table(df_metric, ax=ax, title=displayed_name, scale=1.3)

plot_table(corr_df, ax=axes[-1], title="Pearson r ↑ ∈ [-1, 1]", scale=1.3)

# %%
