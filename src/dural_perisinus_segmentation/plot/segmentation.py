# %%
import matplotlib.pyplot as plt
import pandas as pd
from dural_perisinus_segmentation.plot.utils import (
    boxplot
)

EVALUATION_RATER_1 = "../../../data/nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/bids_pred/desc-SL_label-dante_evaluationDetails.tsv"
EVALUATION_RATER_2 = "../../../data/nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/bids_pred/desc-DD_label-dante_evaluationDetails.tsv"
INTER_RATER_COMPARISON = "../../../data/bids/rater1-SL_rater2-DD_interraterDetails.tsv"

RATER_1_KEY = "test on SLn"
RATER_2_KEY = "test on DD"

#%%
model_evaluation_rater1 = (
    pd.read_csv(
        EVALUATION_RATER_1,
        sep="\t",
    )
    .set_index("participant_id")
    .drop(columns=["session_id"])
)
model_evaluation_rater2 = (
    pd.read_csv(
        EVALUATION_RATER_2,
        sep="\t",
    )
    .set_index("participant_id")
    .drop(columns=["session_id"])
)
inter_rater = (
    pd.read_csv(
        INTER_RATER_COMPARISON,
        sep="\t",
    )
    .set_index("participant_id")
    .drop(columns=["session_id"])
)
df = pd.concat(
    [model_evaluation_rater1, model_evaluation_rater2, inter_rater],
    join="outer",
    keys=[RATER_1_KEY, RATER_2_KEY, "inter-rater"],
    names=["rater"],
    axis=1,
)
df = df.stack(0, future_stack=True).reset_index()
df = df.melt(id_vars=["participant_id", "rater"], value_name="score", var_name="metric")
df = df.sort_values(["participant_id", "metric", "rater"])

# %%
Y_LABEL = "score ↑ ∈ [0, 1]"
HUE_ORDER = [RATER_1_KEY, RATER_2_KEY, "inter-rater"]

df["metric"] = df["metric"].apply(
    lambda x: x.replace("cl_dice", "clDice").replace("dice", "Dice")
)
df = df.rename(columns={"score": Y_LABEL})
metrics = df["metric"].unique()
pairs_hue = [
    (0, 1),  # tested on rater 1  vs  tested on rater 2    → shortest bar
    (1, 2),  # tested on rater 2  vs  inter-rater          → medium bar
    (0, 2),  # tested on rater 1  vs  inter-rater          → longest bar
]
f, ax = boxplot(
    df=df,
    x="metric",
    y=Y_LABEL,
    hue="rater",
    order=metrics,
    hue_order=HUE_ORDER,
    legend_loc="lower center",
    bbox_to_anchor=(0.65, 0),
    test_mode="related",
    pairs_hue=pairs_hue,
    y_space_above_last_value=0.02,
)
plt.tight_layout()
plt.show()

# %%
