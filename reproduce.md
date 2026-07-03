1. Install pixi
2. pixi install
3. dural-perisinus-seg split /Users/thibault.devarax/Desktop/code/neimo/data/bids_icm/labels.tsv --seed 0
4. dural-perisinus-seg bids-to-nnunet /Users/thibault.devarax/Desktop/code/neimo/data/bids_icm '{"suffix": "dante", "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "DD", "label": "dante"}, "data_type": "anat"}' 0 --dataset_name dante
5. ./export_nnunet.sh data
7.
nnUNetv2_plan_and_preprocess -d 0 --verify_dataset_integrity -c 3d_fullres