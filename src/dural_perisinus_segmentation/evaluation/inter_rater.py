from pathlib import Path
from typing import Optional

import click
import pandas as pd
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
    "--rater1_regions_file_type",
    type=BidsFileTypeParam(),
    default=None,
    show_default=True,
    help="A dictionary describing the segmentation per region to get in the BIDS directory for rater 1.",
)
@click.option(
    "--rater2_regions_file_type",
    type=BidsFileTypeParam(),
    default=None,
    show_default=True,
    help="A dictionary describing the segmentation per region to get in the BIDS directory for rater 2.",
)
@click.option(
    "--regions_tsv",
    type=Path,
    default=None,
    show_default=True,
    help="A TSV file describing the region labels. Must contain a column named 'id' with the integer associated to each "
    "ROI in the masks, and a column named 'roi' with the name of the ROI. MANDATORY if rater1_regions_file_type and  rater2_regions_file_type are passed.",
)
@click.option(
    "--rater1_name",
    type=str,
    default="rater1",
    show_default=True,
    help="A dictionary describing the segmentation per region to get in the BIDS directory for rater 1.",
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
    rater1_regions_file_type: Optional[BidsFileType],
    rater2_regions_file_type: Optional[BidsFileType],
    regions_tsv: Optional[Path],
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
        dural-perisinus-seg compute-inter-rater data/my_bids '{"suffix": "mask", "with_entities": {"desc": "rater1", "label": "lymph"}, "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "rater2", "label": "lymph"}, "data_type": "anat"}' --rater1_name XXX --rater2_name YYY
    """
    metrics = MetricsHandler(
        cl_dice=clDiceMetric(pred_key="image", label_key="other_rater"),
        dice=DiceMetric(pred_key="image", label_key="other_rater"),
    )
    volumes_rater1 = MetricsHandler(
        total_volume=VolumeMetric(image_key="image"),
    )
    volumes_rater2 = MetricsHandler(
        total_volume=VolumeMetric(image_key="other_rater"),
    )

    _compute_metrics_and_volumes(
        bids,
        rater1_file_type,
        rater2_file_type,
        metrics,
        volumes_rater1,
        volumes_rater2,
        rater1_name=rater1_name,
        rater2_name=rater2_name,
    )

    if rater1_regions_file_type is not None or rater2_regions_file_type is not None:
        assert rater1_regions_file_type is not None
        assert rater2_regions_file_type is not None

        regions_indices = pd.read_csv(regions_tsv, sep="\t")

        metrics = MetricsHandler()
        volumes_rater1 = MetricsHandler()
        volumes_rater2 = MetricsHandler()

        for _, row in regions_indices.iterrows():
            metrics.add_metrics(
                **{
                    f"cl_dice_{row['roi']}": clDiceMetric(
                        pred_key="image",
                        label_key="other_rater",
                        label=int(row["id"]),
                    ),
                    f"dice_{row['roi']}": DiceMetric(
                        pred_key="image",
                        label_key="other_rater",
                        label=int(row["id"]),
                    ),
                }
            )

            for vol in [volumes_rater1, volumes_rater2]:
                vol.add_metrics(
                    **{
                        row['roi']: VolumeMetric(
                            image_key=mask_name,
                            label=int(row["id"]),
                        )
                        for mask_name in ["image", "other_rater"]
                    }
                )

        _compute_metrics_and_volumes(
            bids,
            rater1_regions_file_type,
            rater2_regions_file_type,
            metrics,
            volumes_rater1,
            volumes_rater2,
            rater1_name=rater1_name,
            rater2_name=rater2_name,
            suffix="Roi",
        )


def _compute_metrics_and_volumes(
    bids: Path,
    mask_rater_1: BidsFileType,
    mask_rater_2: BidsFileType,
    metrics: MetricsHandler,
    volumes_rater1: MetricsHandler,
    volumes_rater2: MetricsHandler,
    rater1_name: str,
    rater2_name: str,
    suffix: Optional[str] = None,
):
    bids_dataset = BidsDataset(
        bids,
        file_type=mask_rater_1,
        masks={"other_rater": mask_rater_2},
    )

    bids_dataset.sanity_check(spatial_checks=["spacing", "shape"])
    loader = DataLoader(bids_dataset, batch_size=2)

    metrics.init_metrics()
    volumes_rater1.init_metrics()
    volumes_rater2.init_metrics()

    desc = f'({suffix})' if suffix else ''

    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(metrics)(masks)
        for masks in tqdm(
            loader,
            total=len(loader),
            desc=f"Computing metrics {desc}",
            unit="batches",
        )
    )
    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(volumes_rater1)(masks)
        for masks in tqdm(
            loader,
            total=len(loader),
            desc=f"Computing volumes for {rater1_name} {desc}",
            unit="batches",
        )
    )
    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(volumes_rater2)(masks)
        for masks in tqdm(
            loader,
            total=len(loader),
            desc=f"Computing volumes for {rater2_name} {desc}",
            unit="batches",
        )
    )

    metrics.aggregate()
    volumes_rater1.aggregate()
    volumes_rater2.aggregate()

    volumes_rater1.save(
        path=bids / f"rater-{rater1_name}_volumes{suffix or ''}.tsv",
        details_path=bids / f"rater-{rater1_name}_volumes{suffix or ''}Details.tsv",
    )
    volumes_rater2.save(
        path=bids / f"rater-{rater2_name}_volumes{suffix or ''}.tsv",
        details_path=bids / f"rater-{rater2_name}_volumes{suffix or ''}Details.tsv",
    )
    metrics.save(
        path=bids
        / f"rater1-{rater1_name}_rater2-{rater2_name}_interrater{suffix or ''}.tsv",
        details_path=bids
        / f"rater1-{rater1_name}_rater2-{rater2_name}_interrater{suffix or ''}Details.tsv",
    )
