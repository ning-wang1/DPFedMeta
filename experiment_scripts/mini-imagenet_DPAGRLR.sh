#!/bin/sh
cd ..
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:

python DPAGRLR.py --name_of_args_json_file experiment_config/mini-imagenet_5_2_0.01_48_5_0_record.json --gpu_to_use 0