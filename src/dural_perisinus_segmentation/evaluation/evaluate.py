from pathlib import Path
from typing import Optional

import click
import pandas as pd
from clinicadl.data.dataloader import DataLoader
from clinicadl.data.datasets import BidsDataset
from clinicadl.io.bids import Bids, BidsFileType
from clinicadl.metrics import MetricsHandler
from clinicadl.transforms import TransformsHandler
from joblib import Parallel, delayed
from tqdm import tqdm

from ..utils import BidsFileTypeParam
from .metrics import DiceMetric, VolumeMetric, clDiceMetric
from .utils import ColorMask


@click.command(no_args_is_help=True)
@click.argument("bids_output", type=Path)
@click.argument("bids_input", type=Path)
@click.argument(
    "mask_file_type",
    type=BidsFileTypeParam(),
    nargs=-1,
)
@click.option(
    "--mask_regions_file_type",
    type=BidsFileTypeParam(),
    default=None,
    show_default=True,
    multiple=True,
    help="A dictionary describing the segmentation masks to get in 'bids_input' to get results by regions. "
    "It must be parameters accepted by clinicadl.io.BidsFileType. Multiple values can be passed (if multiple raters). "
    "If multiple raters, it must be pass in the same order as 'mask_file_type'.",
)
@click.option(
    "--regions_tsv",
    type=Path,
    default=None,
    show_default=True,
    help="A TSV file describing the region labels. Must contain a column named 'id' with the integer associated to each "
    "ROI in the masks, and a column named 'roi' with the name of the ROI. MANDATORY if mask_regions_file_type is passed.",
)
def evaluate(
    bids_output: Path,
    bids_input: Path,
    mask_file_type: tuple[BidsFileType, ...],
    mask_regions_file_type: Optional[tuple[BidsFileType, ...]],
    regions_tsv: Optional[Path],
) -> None:
    """
    To compute the Dice Score and the clDice on the model's predictions.

    Args:\n
        bids_output (Path) : The BIDS directory where the model predictions are saved.\n
        bids_input (Path) : The BIDS directory where the human annotations are saved.\n
        mask_file_type (str) : A dictionary describing the segmentation masks to get in 'bids_input'. It must be parameters accepted by clinicadl.io.BidsFileType. Multiple values can be passed (if multiple raters).

    Example:\n
        dural-perisinus-seg evaluate data/bids_out data/bids_in '{"suffix": "mask", "with_entities": {"desc": "rater1", "label": "lymph"}, "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "rater2", "label": "lymph"}, "data_type": "anat"}' --mask_regions_file_type '{"suffix": "mask", "with_entities": {"desc": "rater1", "label": "lymphRoi"}, "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "rater2", "label": "lymphRoi"}, "data_type": "anat"}'
    """
    metrics = MetricsHandler(
        cldice=clDiceMetric(pred_key="image", label_key="gt"),
        dice=DiceMetric(pred_key="image", label_key="gt"),
    )

    for i, mask in enumerate(mask_file_type, start=1):
        _compute_metrics_and_volumes(
            bids_output,
            bids_input,
            gt_mask=mask,
            metrics=metrics,
            id=i,
        )

    if mask_regions_file_type:
        assert regions_tsv is not None
        regions_indices = pd.read_csv(regions_tsv, sep="\t")

        metrics = MetricsHandler()
        volumes = MetricsHandler()

        for _, row in regions_indices.iterrows():
            metrics.add_metrics(
                **{
                    f"cldice_{row['roi']}": clDiceMetric(
                        pred_key="regions",
                        label_key="gt",
                        label=int(row["id"]),
                    ),
                    f"dice_{row['roi']}": DiceMetric(
                        pred_key="regions",
                        label_key="gt",
                        label=int(row["id"]),
                    ),
                }
            )

            volumes.add_metrics(
                **{
                    row["roi"]: VolumeMetric(
                        image_key="regions",
                        label=int(row["id"]),
                    ),
                }
            )

        for i, mask_region in enumerate(mask_regions_file_type, start=1):
            _compute_metrics_and_volumes(
                bids_output,
                bids_input,
                gt_mask=mask_region,
                metrics=metrics,
                id=i,
                volumes=volumes,
                transforms=TransformsHandler(
                    image_transforms=[
                        ColorMask(
                            mask_key="image",
                            regions_key="gt",
                            colored_mask_name="regions",
                        )
                    ]
                ),
                suffix="Roi",
            )

    bids = BidsDataset(
        bids_output,
        file_type=BidsFileType(data_type="anat", suffix="dseg"),
    )
    bids.sanity_check(spatial_checks=["spacing", "shape"])
    loader = DataLoader(bids, batch_size=2)

    volumes = MetricsHandler(total_volume=VolumeMetric(image_key="image"))
    volumes.init_metrics()
    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(volumes)(preds)
        for preds in tqdm(
            loader, total=len(loader), desc="Computing volumes", unit="batches"
        )
    )
    volumes.aggregate()
    volumes.save(
        path=bids_output / "volumes.tsv",
        details_path=bids_output / "volumesDetails.tsv",
    )


def _compute_metrics_and_volumes(
    bids_output: Path,
    bids_input: Path,
    gt_mask: BidsFileType,
    metrics: MetricsHandler,
    id: int,
    suffix: str = "",
    gt_mask_key: str = "gt",
    transforms: TransformsHandler = TransformsHandler(),
    volumes: Optional[MetricsHandler] = None,
):
    metrics.init_metrics()
    metrics.reset(reset_df=True)
    if volumes is not None:
        volumes.init_metrics()
        volumes.reset(reset_df=True)

    bids = BidsDataset(
        bids_output,
        file_type=BidsFileType(data_type="anat", suffix="dseg"),
        masks={gt_mask_key: (Bids(bids_input), gt_mask)},
        transforms=transforms,
    )
    bids.sanity_check(spatial_checks=["spacing", "shape"])
    loader = DataLoader(bids, batch_size=2)

    desc = f"({suffix})" if suffix else ""

    Parallel(n_jobs=-1, require="sharedmem")(
        delayed(metrics)(preds)
        for preds in tqdm(
            loader,
            total=len(loader),
            desc=f"Computing metrics compared to rater {id} {desc}",
            unit="batches",
        )
    )
    if volumes is not None:
        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(volumes)(preds)
            for preds in tqdm(
                loader,
                total=len(loader),
                desc=f"Computing volumes with rater {id} as reference {desc}",
                unit="batches",
            )
        )

    metrics.aggregate()
    metrics.save(
        path=_infer_tsv_file_path(bids_output, gt_mask, f"evaluation{suffix or ''}"),
        details_path=_infer_tsv_file_path(
            bids_output, gt_mask, f"evaluation{suffix or ''}Details"
        ),
    )

    if volumes is not None:
        volumes.aggregate()
        volumes.save(
            path=_infer_tsv_file_path(
                bids_output, gt_mask, f"volumes{suffix or ''}.tsv"
            ),
            details_path=_infer_tsv_file_path(
                bids_output, gt_mask, f"volumes{suffix or ''}Details.tsv"
            ),
        )


def _infer_tsv_file_path(
    bids_path: Path, mask_file_type: BidsFileType, suffix: str
) -> str:
    bids = Bids(bids_path)

    tsv_file_type = mask_file_type.model_copy()
    tsv_file_type.suffix = suffix
    tsv_file_type.data_type = None
    tsv_file_type.extension = ".tsv"

    return bids.build_path(tsv_file_type)
