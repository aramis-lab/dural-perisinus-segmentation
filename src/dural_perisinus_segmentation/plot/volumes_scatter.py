# A scatter plot of the volumes. Figure 10a.

# %%
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

from dural_perisinus_segmentation.plot.utils import (
    get_aver,
    get_subgroup_pvalues,
    get_ver,
    get_volume_pvalues,
    scatterplots,
)

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
f, ax = scatterplots(
    df,
    plots=[
        (RATER_1_KEY, RATER_2_KEY),
        (RATER_1_KEY, MODEL_KEY),
        (RATER_2_KEY, MODEL_KEY),
    ],
    hue="medical_condition",
    min_=0,
    max_=max_,
    highlight_points=~df.isna().any(axis=1),
    points_legend=["test scan", "training scan"],
    figsize=(15, 5),
    grid_spacing=2,
)
plt.tight_layout()
# %%
get_volume_pvalues(
    df,
    comparisons=[
        ((RATER_1_KEY, RATER_2_KEY), (RATER_1_KEY, MODEL_KEY)),
        ((RATER_1_KEY, RATER_2_KEY), (RATER_2_KEY, MODEL_KEY)),
        ((RATER_1_KEY, MODEL_KEY), (RATER_2_KEY, MODEL_KEY)),
    ],
    quantity="volume",
    mode="related",
)
# %%
df = df.sort_values("automatic model", ascending=True)  # make sure nans are at the end

print("Subgroup analysis")
print("*" * 17 + "\n")
print("volumes")
print("=" * 7)
for col in [RATER_1_KEY, RATER_2_KEY, MODEL_KEY]:
    print(col)
    get_subgroup_pvalues(
        df,
        group_col="medical_condition",
        value_col=col,
    )
    print("-" * 10)
print("\nAVER")
print("=" * 7)
for col1, col2 in combinations([RATER_1_KEY, RATER_2_KEY, MODEL_KEY], 2):
    print(col1, "vs", col2)
    get_subgroup_pvalues(
        df.assign(aver=get_aver(df, col1, col2)[1]),
        group_col="medical_condition",
        value_col="aver",
    )
    print("-" * 10)

print("\nVER")
print("=" * 7)
for col1, col2 in combinations([RATER_1_KEY, RATER_2_KEY, MODEL_KEY], 2):
    print(col1, "vs", col2)
    get_subgroup_pvalues(
        df.assign(ver=get_ver(df, col1, col2)[1]),
        group_col="medical_condition",
        value_col="ver",
    )
    print("-" * 10)
# %%
