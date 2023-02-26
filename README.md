# Optimal Individualized Treatment Rule for Combination Treatment

- Package Requirement
```
numpy		1.22.2
torch			1.11.0a0+git7f4db2c
sklearn		1.0.2
ray			2.2.0
```

- util.py

include helper function, e.g., generate simulation data, estimate treatment-free effects, estimate propensity score

- dem.py

Define the Double Encoder Model and related methods, e.g., fit, predict, evaluate

- tuner.py

Hyper-parameter tuning function

- container.py 

Define the torch input of the DEM

- demo.py

Give a simple example to run the DEM



