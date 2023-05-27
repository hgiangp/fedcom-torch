#!/usr/bin/env bash

tau=$1
dataset=$2
model=$3
learning_rate=$4

./run_main.sh 1 $tau $dataset $model $learning_rate
./run_main.sh 2 $tau $dataset $model $learning_rate
./run_main.sh 3 $tau $dataset $model $learning_rate
./run_main.sh 4 $tau $dataset $model $learning_rate
python3 -u plot_comparison.py --tau=$tau --dataset=$dataset --model=$model --learning_rate=$learning_rate
python3 visualize_image.py |tee logs/visualize_image.log 