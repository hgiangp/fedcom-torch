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
python3 custom_model | tee log_custom_model
``` 
