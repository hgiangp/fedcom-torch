# Generating data 
## Generate data 
### Synthetic dataset 

```
cd data/synthetic/data
rm -r test train 
mkdir train test
cd .. 
python3 generate_synthetic.py | tee generate_synthetic.log 
```

```
cd ../../examples
python3 custom_model.py | tee custom_model.log
``` 
```
python3 network_opt.py | tee network_opt.log
```
```
python3 net_funcs.py | tee net_funcs.log
```
```
python3 server_model.py | tee server_model_train.log
python3 server_model.py | tee server_model_test.log
```
```
python3 client_model.py | tee client_model.log 
```