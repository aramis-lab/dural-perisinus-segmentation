# A scatter plot of the volumes. Figure 10a.

# %%
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

from dural_perisinus_segmentation.plot.utils import (
    get_aver,
    get_subgroup_pvalues,
    get_ver,Œ
    get_volume_pvalues,
    scatterplots,
)

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
