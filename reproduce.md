# To reproduce the experiment

This experiment has been run with the package manager Pixi to ensure a reproducible Python environment.

To reproduce the experiment, follow these steps:

1. [Install Pixi](https://pixi.prefix.dev/latest/installation/)
2. Clone this repo
```bash
git clone https://github.com/aramis-lab/dural-perisinus-segmentation.git && cd dural-perisinus-segmentation
```

3. Install the Python environment
```bash
pixi install
```

4. Organise your data

Your data should be organised following the [BIDS](https://bids.neuroimaging.io/index.html) standard.
Besides, a TSV file with the ids of the participants is required, e.g.:
```text
participant_id  session_id  age sex medical_condition
sub-XXX         ses-M000    35  F   control
sub-YYY         ses-M000    25  F   IIH
```
> **Important**:
> - Even if it is useless here, the column "session_id" is **required**. You can put "ses-M000" everywhere.
> - Columns "age", "sex" and "medical_condition" may be useful to stratify your splits.

5. Split your data into training and test sets using this TSV file and the command `dural-perisinus-seg split`. E.g.:
```bash 
pixi run dural-perisinus-seg split data/bids/metadata.tsv --seed 2
```

6. Export the environment variables required by nnUNet (`nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results`):
```bash
source bash/export_nnunet.sh data
```

7. Convert the BIDS to a format accepted by nnUNet using `dural-perisinus-seg bids-to-nnunet`. E.g.:
```bash
pixi run dural-perisinus-seg bids-to-nnunet data/bids 0 '{"suffix": "dante", "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "DD", "label": "dante"}, "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "SL", "label": "dante"}, "data_type": "anat"}' --dataset_name dante --nnunet_datasets_dir $nnUNet_raw
```

8. Run the nnUNet command `nnUNetv2_plan_and_preprocess` to prepare training:
```bash
pixi run nnUNetv2_plan_and_preprocess -d 0 --verify_dataset_integrity -c 3d_fullres
```

9. Train the model on the 5 folds (0 to 4) with `nnUNetv2_train` (**requires a machine with a GPU**):
```bash
pixi run -e training nnUNetv2_train 0 3d_fullres FOLD --val_best -tr nnUNetTrainer_100epochs -device cuda
```

10. Run inference on the test images with `nnUNetv2_predict`:
```bash
pixi run -e training nnUNetv2_predict -i $nnUNet_raw/Dataset000_dante/imagesTs -o $nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/predictionsTs -d 0 -c 3d_fullres -chk checkpoint_best.pth -tr nnUNetTrainer_100epochs -device cuda
```

11. Convert nnUNet's predictions to a BIDS with `dural-perisinus-seg nnunet-outputs-to-bids`:
```bash
pixi run dural-perisinus-seg nnunet-outputs-to-bids $nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/predictionsTs 0
```

12. Compute the evaluation metrics with `dural-perisinus-seg evaluate`. E.g.:
```bash
pixi run dural-perisinus-seg evaluate $nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/bids_pred data/bids '{"suffix": "mask", "with_entities": {"desc": "DD", "label": "dante"}, "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "SL", "label": "dante"}, "data_type": "anat"}'
```

13. Compute the inter-rater agreement with `dural-perisinus-seg compute-inter-rater`. E.g.:
```bash
pixi run dural-perisinus-seg compute-inter-rater data/bids '{"suffix": "mask", "with_entities": {"desc": "SL", "label": "dante"}, "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "DD", "label": "dante"}, "data_type": "anat"}' --rater1_name SL --rater2_name DD
```

14. Plot the results using `plot.segmentation.py` (Figure 9), `plot.volumes_scatter.py` (Figure 10a) and `plot.volumes_blandaltman.py` (Figure 10b).