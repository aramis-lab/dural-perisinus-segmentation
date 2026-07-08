from pathlib import Path

import click
from clinicadl.data.dataloader import DataLoader
from clinicadl.data.datasets import BidsDataset
from clinicadl.io.bids import Bids, BidsFileType
from clinicadl.metrics import MetricsHandler
from joblib import Parallel, delayed
from tqdm import tqdm

from ..utils import BidsFileTypeParam
from .metrics import DiceMetric, clDiceMetric, VolumeMetric


@click.command(no_args_is_help=True)
@click.argument("bids_output", type=Path)
@click.argument("bids_input", type=Path)
@click.argument(
    "mask_file_type",
    type=BidsFileTypeParam(),
    nargs=-1,
)
def evaluate(
    bids_output: Path, bids_input: Path, mask_file_type: tuple[BidsFileType, ...]
) -> None:
    """
    To compute the Dice Score and the clDice on the model's predictions.

    Args:\n
        bids_output (Path) : The BIDS directory where the model predictions are saved.\n
        bids_input (Path) : The BIDS directory where the human annotations are saved.\n
        mask_file_type (str) : A dictionary describing the segmentation masks to get in 'bids_input'. It must be parameters accepted by clinicadl.io.BidsFileType. Multiple values can be passed (if multiple raters).

    Example:\n
        dural-perisinus-seg evaluate data/bids_out data/bids_in '{"suffix": "mask", "with_entities": {"desc": "rater1", "label": "dante"}, "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "rater2", "label": "dante"}, "data_type": "anat"}'

    """
    metrics = MetricsHandler(
        cl_dice=clDiceMetric(pred_key="image", label_key="gt"),
        dice=DiceMetric(pred_key="image", label_key="gt"),
        pred_volume=VolumeMetric(image_key="image"),
    )
    metrics.init_metrics()
    for mask in mask_file_type:
        metrics.reset(reset_df=True)
        bids = BidsDataset(
            bids_output,
            file_type=BidsFileType(data_type="anat", suffix="dseg"),
            masks={"gt": (bids_input, mask)},
        )
        loader = DataLoader(bids, batch_size=2)
        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(metrics)(preds)
            for preds in tqdm(
                loader, total=len(loader), desc="Computing metrics", unit="batches"
            )
        )

        metrics.aggregate()
        bids_out = Bids(bids_output)

        tsv_file_type = mask.model_copy()
        tsv_file_type.suffix = "evaluation"
        tsv_file_type.data_type = None
        tsv_file_type.extension = ".tsv"
        detailed_tsv_file_type = tsv_file_type.model_copy()
        detailed_tsv_file_type.suffix = "evaluationDetails"
        metrics.save(
            path=bids_out.build_path(tsv_file_type),
            details_path=bids_out.build_path(detailed_tsv_file_type),
        )
