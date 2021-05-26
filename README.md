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
Y, X, A, optA = getdata(sample_size)
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
