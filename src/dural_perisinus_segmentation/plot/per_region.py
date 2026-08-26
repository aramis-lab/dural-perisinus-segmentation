# To plot the results per region.

# %%
from pathlib import Path
import pandas as pd


BIDS_OUT = Path(
    "../../../data/nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/bids_pred/"
)
BIDS_IN = Path("../../../data/bids")

EVALUATION_RATER_1 = BIDS_OUT / "desc-SL_label-lymph_evaluationDetails.tsv"
EVALUATION_RATER_2 = BIDS_OUT / "desc-DD_label-lymph_evaluationDetails.tsv"
EVALUATION_RATER_1_ROI = BIDS_OUT / "desc-SL_label-lymph_evaluationRoiDetails.tsv"
EVALUATION_RATER_2_ROI = BIDS_OUT / "desc-DD_label-lymph_evaluationRoiDetails.tsv"
INTER_RATER_COMPARISON = BIDS_IN / "rater1-SL_rater2-DD_interraterDetails.tsv"
INTER_RATER_COMPARISON_ROI = BIDS_IN / "rater1-SL_rater2-DD_interraterRoiDetails.tsv"

PRED_VOLUMES = BIDS_OUT / "volumesDetails.tsv"
SL_VOLUMES = BIDS_IN / "rater-SL_volumesDetails.tsv"
DD_VOLUMES = BIDS_IN / "rater-DD_volumesDetails.tsv"
PRED_VOLUMES_ROI_SL = BIDS_OUT / "desc-SL_label-lymph_volumesRoiDetails.tsv"
PRED_VOLUMES_ROI_DD = BIDS_OUT / "desc-DD_label-lymph_volumesRoiDetails.tsv"
SL_VOLUMES_ROI = BIDS_IN / "rater-SL_volumesRoiDetails.tsv"
DD_VOLUMES_ROI = BIDS_IN / "rater-DD_volumesRoiDetails.tsv"

# %%
RATER_1_KEY = "SLn"
RATER_2_KEY = "DD"

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

# %%
