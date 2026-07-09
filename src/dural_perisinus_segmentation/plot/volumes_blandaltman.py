# %%
import pandas as pd
from dural_perisinus_segmentation.plot.utils import bland_altman_plots

PRED_VOLUMES = "../../../data/nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/bids_pred/volumesDetails.tsv"
RATERS_VOLUMES = "../../../data/bids/rater1-SL_rater2-DD_volumesDetails.tsv"
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
)
rater_volumes = (
    pd.read_csv(
        RATERS_VOLUMES,
        sep="\t",
    )
    .set_index("participant_id")
    .drop(columns=["session_id"])
)
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

df = df.rename(
    columns={
        RATERS_VOLUMES.split("rater1-")[1].split("_")[0]: RATER_1_KEY,
        RATERS_VOLUMES.split("rater2-")[1].split("_")[0]: RATER_2_KEY,
        "pred_volume": MODEL_KEY,
    }
)

# %%
f, ax = bland_altman_plots(
    df,
    plots=[(RATER_1_KEY, RATER_2_KEY), (RATER_1_KEY, MODEL_KEY), (RATER_2_KEY, MODEL_KEY)],
    hue="medical_condition",
    quantity="volume",
    unit="$cm^3$",
    figsize=(15,4),
    titles=[("DD vs SLn"), ("automatic model vs SLn"), ("automatic model vs DD")],
    grid_spacing=1,
)

# %%
