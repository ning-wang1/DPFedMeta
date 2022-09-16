#!/bin/sh
cd ..
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:

python DPAGRLR.py --name_of_args_json_file experiment_config/omniglot_5_8_0.1_64_5_0_record.json --gpu_to_use 0