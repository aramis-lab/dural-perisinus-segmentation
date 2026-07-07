from __future__ import annotations
import json
from pathlib import Path
import click
from clinicadl.data.datasets import BidsDataset
from clinicadl.io.bids import BidsFileType
from clinicadl.split import SingleSplit
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import ToCanonicalConfig
from tqdm import tqdm
from joblib import Parallel, delayed


class _BidsFileTypeParam(click.ParamType[BidsFileType]):
    def convert(self, value, param, ctx) -> BidsFileType:
        try:
            dict_ = json.loads(value)
            return BidsFileType(**dict_)
        except ValueError:
            self.fail(f"{value!r} is not a valid integer", param, ctx)


NNUNET_RAW = "nnUNet_raw"


@click.command(no_args_is_help=True)
@click.argument(
    "bids_path",
    type=Path,
)
@click.argument(
    "dataset_id",
    type=click.IntRange(min=0),
)
@click.argument(
    "img_file_type",
    type=_BidsFileTypeParam(),
)
@click.argument(
    "mask_file_type",
    type=_BidsFileTypeParam(),
    nargs=-1,
)
@click.option(
    "--dataset_name",
    type=str,
    default="noname",
    show_default=True,
    help="Name to give to the dataset.",
)
@click.option(
    "--split_dir",
    type=Path,
    default=Path("data") / "split",
    show_default=True,
    help="The directory where to find the training/test split (produced with the 'split' command).",
)
@click.option(
    "--nnunet_datasets_dir",
    type=Path,
    default=Path("data") / NNUNET_RAW,
    show_default=True,
    help="The directory where to save the dataset in the nnUNet format.",
)
def bids_to_nnunet(
    bids_path: Path,
    dataset_id: int,
    img_file_type: BidsFileType,
    mask_file_type: tuple[BidsFileType, ...],
    dataset_name: str,
    split_dir: Path,
    nnunet_datasets_dir: Path,
) -> None:
    """
    Convert a BIDS to the dataset format accepted by nnUNet.

    The dataset will be saved in <nnunet_datasets_dir>/Dataset<dataset_id>_<dataset_name>.

    Args:\n
        bids_path (Path) : Path to the BIDS directory.\n
        dataset_id (int) : The id to give to the dataset.\n
        img_file_type (str) : A dictionary describing the images to get in the BIDS directory. It must be parameters accepted by clinicadl.io.BidsFileType.\n
        mask_file_type (str) : A dictionary describing the segmentation masks to get in the BIDS directory. It must be parameters accepted by clinicadl.io.BidsFileType. Multiple values can be passed (if multiple raters).

    Example:\n
        dural-perisinus-seg bids-to-nnunet data/my_bids '{"suffix": "dante", "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "rater1"}, "data_type": "anat"}' 0
    """
    cnt_training = 0
    subjects_ids = {}
    for mask in mask_file_type:
        print("Exporting data for mask: ", {str(mask)})
        bids = BidsDataset(
            bids_path,
            file_type=img_file_type,
            masks={"gt": mask},
            transforms=TransformsHandler(image_transforms=[ToCanonicalConfig()]),
        )
        bids.sanity_check(spatial_checks=["affine"])

        split = SingleSplit(split_dir).get_split(bids)

        dataset_path = (
            Path(nnunet_datasets_dir) / f"Dataset{dataset_id:03}_{dataset_name}"
        )
        (dataset_path / "imagesTr").mkdir(exist_ok=True, parents=True)
        (dataset_path / "imagesTs").mkdir(exist_ok=True, parents=True)
        (dataset_path / "labelsTr").mkdir(exist_ok=True, parents=True)

        train_results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_save_train_sample)(
                cnt_training + k,
                data,
                dataset_path,
            )
            for k, data in tqdm(
                enumerate(split.train_dataset),
                total=len(split.train_dataset),
                desc="Exporting training images",
            )
        )

        subjects_ids.update(dict(train_results))
        cnt_training += len(train_results)

    bids = BidsDataset(
        bids_path,
        file_type=img_file_type,
        transforms=TransformsHandler(image_transforms=[ToCanonicalConfig()]),
    )
    split = SingleSplit(split_dir).get_split(bids)

    test_results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_save_test_sample)(
            cnt_training + k,
            data,
            dataset_path,
        )
        for k, data in tqdm(
            enumerate(split.val_dataset),
            total=len(split.val_dataset),
            desc="Exporting test images",
        )
    )

    subjects_ids.update(dict(test_results))

    dataset_description = {
        "channel_names": {"0": "T1"},
        "labels": {
            "background": 0,
            "perisinusal_fluid": 1,
        },
        "numTraining": cnt_training,
        "file_ending": ".nii.gz",
    }

    with open(dataset_path / "dataset.json", "w+") as f:
        json.dump(dataset_description, f, indent=4)

    with open(dataset_path / "subject_ids.json", "w+") as f:
        json.dump(subjects_ids, f, indent=4)


def _save_train_sample(i, data, dataset_path):
    img_dest_path = dataset_path / "imagesTr" / f"sub_{i:03}_0000.nii.gz"
    seg_dest_path = dataset_path / "labelsTr" / f"sub_{i:03}.nii.gz"

    data.image.save(img_dest_path)
    data.gt.save(seg_dest_path)

    return f"{i:03}", data.participant_id


def _save_test_sample(j, data, dataset_path):
    img_dest_path = dataset_path / "imagesTs" / f"sub_{j:03}_0000.nii.gz"

    data.image.save(img_dest_path)

    return f"{j:03}", data.participant_id
