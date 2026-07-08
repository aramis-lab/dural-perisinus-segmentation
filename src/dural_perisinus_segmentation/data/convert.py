from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile

import click
from clinicadl.data.datasets import BidsDataset
from clinicadl.io.bids import BidsFileType
from clinicadl.split import SingleSplit
from clinicadl.transforms import TransformsHandler
from clinicadl.transforms.config import ToCanonicalConfig
from joblib import Parallel, delayed
from tqdm import tqdm

from ..utils import BidsFileTypeParam

NNUNET_RAW = "nnUNet_raw"
NNUNET_RESULTS = "nnUNet_results"


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
    type=BidsFileTypeParam(),
)
@click.argument(
    "mask_file_type",
    type=BidsFileTypeParam(),
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
        dural-perisinus-seg bids-to-nnunet data/my_bids '{"suffix": "dante", "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "rater1"}, "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "rater2"}, "data_type": "anat"}'
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


@click.command(no_args_is_help=True)
@click.argument(
    "output_dir",
    type=Path,
)
@click.argument(
    "dataset_id",
    type=int,
)
@click.option(
    "--bids_output_name",
    type=str,
    default="bids_pred",
    show_default=True,
    help="The name to give to the output BIDS.",
)
@click.option(
    "--nnunet_raw_data_dir",
    type=Path,
    default=Path("data") / NNUNET_RAW,
    show_default=True,
    help="The nnUNet_raw directory.",
)
def nnunet_outputs_to_bids(
    output_dir: Path,
    dataset_id: int,
    bids_output_name: str,
    nnunet_raw_data_dir: Path,
) -> None:
    """
    Convert the outputs of 'nnUNetv2_predict' to a BIDS directory.

    Args:\n
        output_dir (Path) : The path to the predictions of nnUNet.
        dataset_id (int) : The id given to the dataset (in nnUNet_raw).

    Example:\n
        dural-perisinus-seg nnunet-outputs-to-bids data/nnUNet_results/Dataset000_dante/nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres/predictionsTs 0
    """
    bids_output = output_dir.parent / bids_output_name

    for dataset in nnunet_raw_data_dir.glob("Dataset*"):
        id_ = int(dataset.name.split("_")[0].replace("Dataset", ""))
        if id_ == dataset_id:
            break

    with open(dataset / "subject_ids.json", "r") as f:
        ids = json.load(f)

    for pred in output_dir.glob("*.nii.gz"):
        id_ = pred.stem.replace("sub_", "").split(".")[0]
        participant = ids[id_]
        dest_dir = bids_output / f"{participant}" / "ses-M000" / "anat"
        dest_dir.mkdir(exist_ok=True, parents=True)
        copyfile(
            pred,
            dest_dir / f"{participant}_ses-M000_dseg.nii.gz",
        )

    with open(bids_output / "dataset_description.json", "w") as f:
        json.dump(
            {
                "Name": "Predictions from nnUNet",
                "BIDSVersion": "1.11.0",
                "DatasetType": "derivative",
            },
            f,
        )
