#!/bin/bash

INPUT_PATH=$(realpath "$1")

export nnUNet_raw="$INPUT_PATH/nnUNet_raw"
export nnUNet_preprocessed="$INPUT_PATH/nnUNet_preprocessed"
export nnUNet_results="$INPUT_PATH/nnUNet_results"

echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"