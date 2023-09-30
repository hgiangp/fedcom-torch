# Energy-Efficient Federated Learning-enabled Digital Twin for UAV-aided Vehicular Networks 
This repository implements all the experiments in the paper 
> [Energy-Efficient Federated Learning (FL)-enabled Digital Twin (DT) for UAV-aided Vehicular Networks](https://drive.google.com/file/d/1UBU8bYQHr2nrY5Qr7OyiOp6gEJwR7Aye/view?usp=sharing) 

Federated Learning-enabled Digital Twin has recently attracted research attention to enable applications of intelligent transportation systems. Considering the accuracy and latency requirements of DT applications under the vehicle mobility impact, we propose a dynamic optimization framework to minimize energy consumption given the accuracy and latency constraints. 

## General Guideline 
The repository includes two main parts: Federated Learning and Network Optimization. 
- Federated Learning is implemented by PyTorch. We use the multinomial logistic regression (mclr) with stochastic gradient descent (SGD) as the FL optimization model. The models are implemented in `./flean/models` folder. 
- For the network optimization part, convex optimization techniques, including the Dinkelbach method and primal-dual interior-point method are implemented. You can find the interior-point method implementation from scratch in [network_optim.py](./src/network_optim.py) or implementation by `minimize` library from `Scipy.optimize` package in [optimization_scipy.py](./src/optimization_scipy.py). 

## Data Preparation 
We utilize 3 datasets including Synthetic, MNIST and CIFAR10. The data will be generated into `./data` folder accordingly. To generate data for the MNIST dataset in either iid or niid manner, the following commands can be run: 
```
python3 ./data/mnist/generate_niid.py | tee ./data/mnist/generate_niid.log
python3 ./data/mnist/generate_iid.py | tee ./data/mnist/generate_iid.log
```

## Repository Structure
The main source codes are implemented in `./src` folder. 
- [main.py](main.py): The arguments are read to construct the system. 
- [system_model.py](./src/system_model.py): The system model, where the FL model and network optimization model are initialized. 
- [server_model.py](./src/server_model.py): The server model initializes the clients and communicates with clients during the FL process.  
- [client_model.py](./src/client_model.py): The client model, where the FL model parameters are trained on client's dataset during the FL process.  
- [location_model.py](./src/location_model.py): The location model of clients, where the location are updated during the FL process.
- [parse_log.py](parse_log.py): By `tee`, log file will be written into `./logs` folder. 

## Running the Simulation 
- (1) Set the simulation parameters in [run.sh](run.sh) file. Refer to [main.py](main.py) for description of the simulation parameters. 
- (2) Run the simulation and plot the simulation results by `./run.sh`

## Visualize the MNIST Dataset 
After running the simulation, the FL model are saved into `./models` folder. To visualize the MNIST images and SNE, we can run 
```
python3 visualize_image.py |tee logs/visualize_image.log 
```