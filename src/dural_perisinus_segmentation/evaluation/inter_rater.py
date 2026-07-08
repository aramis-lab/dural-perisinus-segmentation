from pathlib import Path

import click
from clinicadl.data.dataloader import DataLoader
from clinicadl.data.datasets import BidsDataset
from clinicadl.io.bids import BidsFileType
from clinicadl.metrics import MetricsHandler
from joblib import Parallel, delayed
from tqdm import tqdm

from ..utils import BidsFileTypeParam
from .metrics import DiceMetric, VolumeMetric, clDiceMetric


@click.command(no_args_is_help=True)
@click.argument("bids", type=Path)
@click.argument(
    "rater1_file_type",
    type=BidsFileTypeParam(),
)
@click.argument(
    "rater2_file_type",
    type=BidsFileTypeParam(),
)
@click.option(
    "--rater1_name",
    type=str,
    default="rater1",
    show_default=True,
    help="The name to give to rater 1.",
)
@click.option(
    "--rater2_name",
    type=str,
    default="rater2",
    show_default=True,
    help="The name to give to rater 2.",
)
def compute_inter_rater(
    bids: Path,
    rater1_file_type: BidsFileType,
    rater2_file_type: BidsFileType,
    rater1_name: str,
    rater2_name: str,
) -> None:
    """
    Computes inter-rater Dice Score and clDice, as well as the volumes.

    Args:\n
        bids (Path) : The BIDS directory where the annotations are saved.\n
        rater1_file_type (str) : A dictionary describing the segmentation masks to get in the BIDS directory for rater 1.\n
        rater2_file_type (str) : A dictionary describing the segmentation masks to get in the BIDS directory for rater 2.

    Example:\n
        dural-perisinus-seg compute-inter-rater data/my_bids '{"suffix": "mask", "with_entities": {"desc": "rater1", "label": "dante"}, "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "rater2", "label": "dante"}, "data_type": "anat"}' --rater1_name XXX --rater2_name YYY
    """
    metrics = MetricsHandler(
        cl_dice=clDiceMetric(pred_key="image", label_key="other_rater"),
        dice=DiceMetric(pred_key="image", label_key="other_rater"),
    )
    volumes = MetricsHandler(
        **{
            rater1_name: VolumeMetric(image_key="image"),
            rater2_name: VolumeMetric(image_key="other_rater"),
        },
    )
    metrics.init_metrics()
    volumes.init_metrics()

    bids_dataset = BidsDataset(
        bids,
        file_type=rater1_file_type,
        masks={"other_rater": rater2_file_type},
    )
    loader = DataLoader(bids_dataset, batch_size=2)

    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(metrics)(masks)
        for masks in tqdm(
            loader, total=len(loader), desc="Computing metrics", unit="batches"
        )
    )
    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(volumes)(masks)
        for masks in tqdm(
            loader, total=len(loader), desc="Computing volumes", unit="batches"
        )
    )

    metrics.aggregate()
    volumes.aggregate()

    volumes.save(
        path=bids / f"rater1-{rater1_name}_rater2-{rater2_name}_volumes.tsv",
        details_path=bids
        / f"rater1-{rater1_name}_rater2-{rater2_name}_volumesDetails.tsv",
    )
    metrics.save(
        path=bids / f"rater1-{rater1_name}_rater2-{rater2_name}_interrater.tsv",
        details_path=bids / f"rater1-{rater1_name}_rater2-{rater2_name}_interraterDetails.tsv",
    )
