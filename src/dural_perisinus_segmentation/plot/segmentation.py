# To plot the segmentation metrics (DSC and clDice). Figure 9.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from dural_perisinus_segmentation.plot.utils import (
    boxplot,
    get_mean_and_conf,
    get_subgroup_pvalues,
)

EVALUATION_RATER_1 = "../../../data/nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/bids_pred/desc-SL_label-dante_evaluationDetails.tsv"
EVALUATION_RATER_2 = "../../../data/nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/bids_pred/desc-DD_label-dante_evaluationDetails.tsv"
INTER_RATER_COMPARISON = "../../../data/bids/rater1-SL_rater2-DD_interraterDetails.tsv"
METADATA = "../../../data/bids/metadata.tsv"

RATER_1_KEY = "test on SLn"
RATER_2_KEY = "test on DD"

# %%
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
metadata = (
    pd.read_csv(
        METADATA,
        sep="\t",
    )
)[["participant_id", "medical_condition"]]
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
df = df.merge(metadata, how="left")

# %%
Y_LABEL = "score ↑ ∈ [0, 1]"
HUE_ORDER = [RATER_1_KEY, RATER_2_KEY, "inter-rater"]

df["metric"] = df["metric"].apply(
    lambda x: x.replace("cl_dice", "clDice").replace("dice", "DSC")
)
df = df.rename(columns={"score": Y_LABEL})
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
    order=["DSC", "clDice"],
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
print("summary")
print("-------")
print("inter-rater")
print(
    " DSC: ",
    get_mean_and_conf(
        df[(df["rater"] == "inter-rater") & (df["metric"] == "DSC")][Y_LABEL]
    ),
)
print(
    " clDice: ",
    get_mean_and_conf(
        df[(df["rater"] == "inter-rater") & (df["metric"] == "clDice")][Y_LABEL]
    ),
)
print(RATER_1_KEY)
print(
    " DSC: ",
    get_mean_and_conf(
        df[(df["rater"] == RATER_1_KEY) & (df["metric"] == "DSC")][Y_LABEL]
    ),
)
print(
    " clDice: ",
    get_mean_and_conf(
        df[(df["rater"] == RATER_1_KEY) & (df["metric"] == "clDice")][Y_LABEL]
    ),
)
print(RATER_2_KEY)
print(
    " DSC: ",
    get_mean_and_conf(
        df[(df["rater"] == RATER_2_KEY) & (df["metric"] == "DSC")][Y_LABEL]
    ),
)
print(
    " clDice: ",
    get_mean_and_conf(
        df[(df["rater"] == RATER_2_KEY) & (df["metric"] == "clDice")][Y_LABEL]
    ),
)

# %%
print("Subgroup analysis")
print("*" * 17)
for metric in ["DSC", "clDice"]:
    print("\n", metric)
    print("=" * 7)
    for rater in [RATER_1_KEY, RATER_2_KEY, "inter-rater"]:
        print(rater)
        get_subgroup_pvalues(
            df[(df["rater"] == rater) & (df["metric"] == metric)],
            group_col="medical_condition",
            value_col=Y_LABEL,
        )
        print("-" * 10)
# %%
