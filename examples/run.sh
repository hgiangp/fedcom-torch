#!/usr/bin/env bash

./run_system.sh 1 50
./run_system.sh 2 50
./run_system.sh 3 50
./run_system.sh 4 50
python3 -u plot_comparison.py