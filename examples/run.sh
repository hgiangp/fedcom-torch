#!/usr/bin/env bash

tau=35

./run_system.sh 1 $tau
./run_system.sh 2 $tau
./run_system.sh 3 $tau
./run_system.sh 4 $tau
python3 -u plot_comparison.py