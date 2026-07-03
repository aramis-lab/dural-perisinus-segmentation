1. Install pixi
2. pixi install
3. dural_perisinus_seg split /Users/thibault.devarax/Desktop/code/neimo/data/bids_icm/labels.tsv --seed 0
4. dural_perisinus_seg bids-to-nnunet /Users/thibault.devarax/Desktop/code/neimo/data/bids_icm '{"suffix": "dante", "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "DD", "label": "dante"}, "data_type": "anat"}' 0 --dataset_name dante
5.
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
6.
echo "$nnUNet_raw"
echo "$nnUNet_preprocessed"
echo "$nnUNet_results"
7.
nnUNetv2_plan_and_preprocess -d 0 --verify_dataset_integrity -c 3d_fullres