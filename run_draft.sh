#!/usr/bin/env bash

# python3 ./flearn/models/synthetic/mclr.py| tee logs/mclr.log
# python3 client_model.py | tee logs/client_model.log
# python3 -u main.py --sce_idx=$1 --tau=$2 --dataset=$3 --model=$4| tee logs/main.log
# ./run.sh 40 mnist mclr 
./run_main.sh 4 40 mnist mclr 