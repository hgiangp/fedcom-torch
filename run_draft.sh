#!/usr/bin/env bash

# python3 ./flearn/models/synthetic/mclr.py| tee logs/mclr.log
# python3 client_model.py | tee logs/client_model.log
# python3 -u main.py --sce_idx=$1 --tau=$2 --dataset=$3 --model=$4 --learning_rate=$5| tee logs/main.log
# python3 visualize_image.py |tee logs/visualize_image.log 
# python3 ./data/mnist/generate_niid.py | tee ./data/mnist/generate_niid.log
# python3 ./data/emnist/generate_niid.py | tee ./data/emnist/generate_niid.log

# python -m torch.utils.bottleneck main.py --sce_idx=$sce_idx --tau=$tau --dataset=$dataset --model=$model --learning_rate=$learning_rate \
    # | tee logs/$dataset/s$sce_idx/system_model.log

# ./run_main.sh 4 40 mnist mclr 0.01
./run.sh 40 mnist mclr 0.01

# sce_idx=4
# tau=40
# dataset=mnist 
# model=mclr 
# learning_rate=0.01

# python3 -u plot_comparison.py --tau=$tau --dataset=$dataset --model=$model --learning_rate=$learning_rate