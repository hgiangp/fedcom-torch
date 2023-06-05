#!/usr/bin/env bash

# python3 ./flearn/models/synthetic/mclr.py| tee logs/mclr.log
# python3 client_model.py | tee logs/client_model.log
# python3 -u main.py --sce_idx=$1 --tau=$2 --dataset=$3 --model=$4 --learning_rate=$5| tee logs/main.log
# python3 visualize_image.py |tee logs/visualize_image.log 
# python3 ./data/mnist/generate_niid.py | tee ./data/mnist/generate_niid.log
# python3 ./data/emnist/generate_niid.py | tee ./data/emnist/generate_niid.log

# python -m torch.utils.bottleneck main.py --sce_idx=$sce_idx --tau=$tau --dataset=$dataset --model=$model --learning_rate=$learning_rate \
    # | tee logs/$dataset/s$sce_idx/system_model.log

# ./run_main.sh 4 8 mnist mclr 0.01 True
# ./run.sh 8 mnist mclr 0.01 True

# python3 -u plot_comparison.py --tau=$tau --dataset=$dataset --model=$model --learning_rate=$learning_rate
# python3 plot_synthetic.py --sce_idx=$sce_idx --tau=$tau --dataset=$dataset --model=$model --learning_rate=$learning_rate
# ./run_main.sh 4 100 cifar10 mclr 0.001

# python3 -u main.py --sce_idx=$sce_idx --tau=$tau --dataset=$dataset --model=$model --learning_rate=$learning_rate| tee logs/$dataset/s$sce_idx/system_model_unoptim.log
sce_idx=4
tau=5.82
dataset=mnist 
model=mclr 
learning_rate=0.01
optim=1
gamma=2.0
C_n=0.05
xi_factor=1
for tau in 5.82
do
    for optim in 1 2 3 4
    do
        log_file=logs/$dataset/s$sce_idx/system_model_tau"$tau"_gamma"$gamma"_cn"$C_n"_optim"$optim".log
        python3 -u main.py --sce_idx=$sce_idx --tau=$tau --dataset=$dataset --model=$model \
                            --learning_rate=$learning_rate --optim=$optim \
                            --xi_factor=$xi_factor --C_n=$C_n | tee $log_file
    done
done

# python3 -u plot_comparison.py --tau=$tau --dataset=$dataset --model=$model --learning_rate=$learning_rate --xi_factor=$xi_factor