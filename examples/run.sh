#!/usr/bin/env bash

python3 system_model.py| tee logs/system_model.log
python3 server_model.py| tee logs/server_model.log 
python3 plot_synthetic.py| tee logs/plot_synthetic.log 