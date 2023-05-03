#!/usr/bin/env bash

python3 -u system_model.py --sce_idx=$1 | tee logs/s$1/system_model.log
python3 -u plot_synthetic.py --sce_idx=$1