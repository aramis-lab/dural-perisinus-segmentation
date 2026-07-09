# Deep Learning-Based 3T MRI Segmentation of Dural Perisinus for Assessment of Human Meningeal Lymphatics

This repository is associated to the paper:
**PAPER**

The automatic segmentation model was trained using the [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework.

## Use

The trained **weights** can be found here: **LINK**

You can use our models to **segment your own data** by following [this guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how-to/run-inference.md).

Or you can **fine-tune** them using [this guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/pretraining_and_finetuning.md).

## Reproduce

To reproduce the experiment, follow the [reproducibility guide](reproduce.md).

## Technical details

The configuration used can be found here: **LINK**

Since nnUNet does not support multiple annotations for a single image, we created two copies of each image, one paired with the mask from rater A and the other with the mask from rater B. Because nnUNet uses a fixed number of training iterations per epoch,  this duplication does not affect the epoch length Besides, as images are selected randomly at each iteration, duplicating them does not change the sampling probability of any given image. The only difference with the standard configuration is that the same image may appear more than once within a single batch. With a batch size of 2 and a training dataset of 21 images, this overlap is expected to occur approximately once every 41 batches.

