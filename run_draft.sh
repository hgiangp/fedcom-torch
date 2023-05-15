#!/usr/bin/env bash

# python3 ./flearn/models/synthetic/mclr.py| tee logs/mclr.log
# python3 client_model.py | tee logs/client_model.log
# python3 -u main.py --sce_idx=$1 --tau=$2 --dataset=$3 --model=$4 --learning_rate=$5| tee logs/main.log
# mkdir ./figures/mnist
# cd ./figures/mnist
# mkdir s1 s2 s3 s4 
# mkdir comparison
# cd ../..
# python3 visualize_image.py |tee logs/visualize_image.log 
# python3 ./data/mnist/generate_niid.py | tee ./data/mnist/generate_niid.log
# ./run_main.sh 4 80 mnist mclr 0.01
./run.sh 100 mnist mclr 0.01

# python3 ./data/emnist/generate_niid.py | tee ./data/emnist/generate_niid.log