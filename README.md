# MCITR

## Python implementation of Multi-Channel Individualized Treatment Rule

- Package Requirement
```
numpy		version 1.19.2
autograd	version 1.3
pymanopt	version 0.2.5
torch		version 1.7.1+cu101
sklearn		version 0.0
matplotlib	version 3.4.1
```

- Demo
```
from mcitr from MCITR
import numpy as np
import matplotlib.pyplot as plt
import torch

Y, X, A, optA = getdata(sample_size)

mcitr = MCITR()
history = mcitr.fit(Y, X, A, scenario="ct", verbose=1, device="cpu", epochs=100)

D = mcitr.predict(X, A) # estimated MCITR

value, accuracy = mcitr.evaluate(Y, A, D, optA=optA) # value function and accuracy of estimated MCITR

D_bc = mcitr.realign(X, A, cost, budget, budget_level="population") # estimated MCITR after rotation to satisfy budget constraint
```

- Done
- [X] Estimate optimal MCITR using embedding inner-product networks
- [X] Estimate constrained MCITR with rotational framework

- To Do List
- [ ] Stabilize small weights for propensity scores
- [ ] Doubly robust estimator
- [ ] Speed up population-level budget constraint
- [ ] Theory:
	- [ ] Statistical Optimality
	- [ ] Equivalence between proposed rotational framework and discrete problem
