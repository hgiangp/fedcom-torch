# Generating data 
## Generate data 
### Synthetic dataset 

```
cd data/synthetic
mkdir {train,test}
python3 generate_synthetic.py | tee log_synthetic 
```

```
cd ../../examples
python3 custom_model.py | tee log_custom_model
``` 
```
python3 network_opt.py | tee log_network_opt
```