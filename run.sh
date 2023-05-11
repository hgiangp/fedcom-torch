#!/usr/bin/env bash

tau=$1
dataset=$2
model=$3

./run_main.sh 1 $tau $dataset $model
./run_main.sh 2 $tau $dataset $model
./run_main.sh 3 $tau $dataset $model
./run_main.sh 4 $tau $dataset $model
python3 -u plot_comparison.py