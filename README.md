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

Y, X, A, optA = getdata(200, case=2, seed=1)

mcitr = MCITR(depth_trt=3, depth_cov=3, width_trt=50, width_cov=50, width_embed=3)
history = mcitr.fit(Y, X, A, device="cpu", verbose=1, epochs=50, learning_rate=5e-2)

D = mcitr.predict(X, A) # unconstrained
accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

print("----accuracy: {0}----".format(accuracy))
print("----value: {0}----".format(value))

cost = np.array([0, 0, 0, 0, 1, 1, 1, 1])
budgets = 20

D = mcitr.realign_mckp(X, A, cost = cost, budget=budgets) # constrained

accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

print("----accuracy: {0}----".format(accuracy))
print("----value: {0}----".format(value))
```

- Done
- [X] Estimate optimal MCITR using embedding inner-product networks
- [X] Estimate constrained MCITR with multi-choice knapsack (MCKP)

- To Do List
- [ ] Feature selection for high-dimensional covariate
- [ ] Stabilize small weights for propensity scores
- [ ] Doubly robust estimator
- [ ] Theory:
	- [ ] Statistical Optimality
