# A Bland-Altman plot of the volumes. Figure 6b.

# %%
import pandas as pd

from dural_perisinus_segmentation.plot.utils import bland_altman_plots

PRED_VOLUMES = "../../../data/nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/bids_pred/volumesDetails.tsv"
RATER1_VOLUMES = "../../../data/bids/rater-SL_volumesDetails.tsv"
RATER2_VOLUMES = "../../../data/bids/rater-DD_volumesDetails.tsv"
METADATA = "../../../data/bids/metadata.tsv"

RATER_1_KEY = "SLn"
RATER_2_KEY = "DD"
MODEL_KEY = "automatic model"

# %%
pred_volumes = (
    pd.read_csv(
        PRED_VOLUMES,
        sep="\t",
    )
    .set_index("participant_id")
    .drop(columns=["session_id"])
).rename(columns={"total_volume": MODEL_KEY})
rater_volumes = pd.concat(
    [
        pd.read_csv(
            RATER1_VOLUMES,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
        pd.read_csv(
            RATER2_VOLUMES,
            sep="\t",
        )
        .set_index("participant_id")
        .drop(columns=["session_id"]),
    ],
    axis=1,
    keys=[RATER_1_KEY, RATER_2_KEY],
).droplevel(axis=1, level=1)
metadata = (
    pd.read_csv(
        METADATA,
        sep="\t",
    ).set_index("participant_id")
)[["medical_condition"]]
df = pd.concat(
    [pred_volumes, rater_volumes],
    join="outer",
    axis=1,
)

df /= 1000  # cm3
min_ = df.min()
max_ = df.max()

df = pd.concat(
    [df, metadata],
    join="inner",
    axis=1,
)

# %%
bland_altman_plots(
    df,
    plots=[
        (RATER_1_KEY, RATER_2_KEY),
        (RATER_1_KEY, MODEL_KEY),
        (RATER_2_KEY, MODEL_KEY),
    ],
    hue="medical_condition",
    quantity="volume",
    unit="$cm^3$",
    highlight_points=~df.isna().any(axis=1),
    points_legend=["test scan", "training scan"],
    figsize=(15, 4),
    titles=[("DD vs SLn"), ("automatic model vs SLn"), ("automatic model vs DD")],
    grid_spacing=1,
)

# %%
