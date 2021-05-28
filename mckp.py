from mip import *
import numpy as np


def _mckp(trt_panel, cost, budget):

    n_samples = trt_panel.shape[0]
    n_combinations = trt_panel.shape[1]

    _cost = np.tile(cost, n_samples)

    m = Model(sense=MAXIMIZE)

    x = [m.add_var(var_type=BINARY) for i in range(n_samples * n_combinations)]

    # budget constraints
    
    m += xsum(_cost[i] * x[i] for i in range(n_samples * n_combinations)) <= budget

    # each individual can only accept one treatment

    for i in range(n_samples):

        m += xsum(x[i] for i in range(i * n_combinations, (i + 1) * n_combinations)) == 1

    value = np.reshape(trt_panel, n_samples * n_combinations)

    m.objective = xsum(value[i] * x[i] for i in range(n_samples * n_combinations))

    m.verbose = 0
    status = m.optimize(max_seconds=500)

    print(status)

    sol = np.array([v.x for v in m.vars])
    sol = np.reshape(sol, (n_samples, n_combinations))

    return sol

