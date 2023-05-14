#!/usr/bin/env bash

python3 -u main.py --sce_idx=$1 --tau=$2 --dataset=$3 --model=$4 --learning_rate=$5| tee logs/$3/s$1/system_model.log
python3 plot_synthetic.py --sce_idx=$1 --dataset=$3