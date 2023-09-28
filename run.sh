#!/usr/bin/env bash

sce_idx=4
tau=30
dataset=mnist 
model=mclr 
learning_rate=0.0017
optim=1
gamma=100
C_n=0.7 # 0.2 
xi_factor=1
velocity=90
for sce_idx in 4
do
    for velocity in 40
    do
        log_file=logs/$dataset/s$sce_idx/tau"$tau"_gamma"$gamma"_cn"$C_n"_vec"$velocity"_optim"$optim".log
        python3 -u main.py --sce_idx=$sce_idx --tau=$tau --dataset=$dataset --model=$model \
                            --learning_rate=$learning_rate --optim=$optim \
                            --xi_factor=$xi_factor --C_n=$C_n --velocity=$velocity\
                            | tee $log_file
    done
done

python3 -u plot_comparison.py --tau=$tau --dataset=$dataset --model=$model --learning_rate=$learning_rate --xi_factor=$xi_factor
# python3 plot_synthetic.py --sce_idx=$sce_idx --tau=$tau --dataset=$dataset --model=$model --learning_rate=$learning_rate