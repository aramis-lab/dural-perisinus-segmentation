1. Install pixi
2. pixi install
3. pixi run dural-perisinus-seg split /Users/thibault.devarax/Desktop/code/neimo/data/bids_icm/labels.tsv --seed 2
4. pixi run dural-perisinus-seg bids-to-nnunet /Users/thibault.devarax/Desktop/code/neimo/data/bids_icm 0 '{"suffix": "dante", "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "DD", "label": "dante"}, "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "SL", "label": "dante"}, "data_type": "anat"}' --dataset_name dante
5. source bash/export_nnunet.sh data
6. pixi run nnUNetv2_plan_and_preprocess -d 0 --verify_dataset_integrity -c 3d_fullres
7. pixi run -e training nnUNetv2_train 0 3d_fullres FOLD --val_best -tr nnUNetTrainer_100epochs -device cuda
8. pixi run -e training nnUNetv2_predict -i $nnUNet_raw/Dataset000_dante/imagesTs -o $nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/predictionsTs -d 0 -c 3d_fullres -chk checkpoint_best.pth -tr nnUNetTrainer_100epochs -device cuda
9. pixi run dural-perisinus-seg nnunet-outputs-to-bids /Users/thibault.devarax/Desktop/code/dural-perisinus-segmentation/data/nnUNet_results/Dataset000_dante/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/predictionsTs 0
10. dural-perisinus-seg evaluate /Users/thibault.devarax/Desktop/code/dural-perisinus-segmentation/data_/nnUNet_results/Dataset000_dante/nnUNetTrainer_1epoch__nnUNetPlans__3d_fullres/bids_out /Users/thibault.devarax/Desktop/code/dural-perisinus-segmentation/data_/bids_icm '{"suffix": "mask", "with_entities": {"desc": "DD", "label": "dante"}, "data_type": "anat"}' '{"suffix": "mask", "with_entities": {"desc": "SL", "label": "dante"}, "data_type": "anat"}'