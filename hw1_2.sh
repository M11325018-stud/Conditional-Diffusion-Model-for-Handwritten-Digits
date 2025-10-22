#!/bin/bash

# TODO - run your inference Python3 code
python3 ./src/inference_hw1_2.py "$1" "$2" --ckpt ./hw1_2_segnet_best.pth --tta
