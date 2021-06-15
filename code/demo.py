

from util import getdata
from mcitr import MCITR
import numpy as np


## demo

def demo():

    Y, X, A, optA = getdata(500, case=1, seed=1, family="bernoulli")

    mcitr = MCITR(depth_trt=3, depth_cov=3, depth_men=3, width_trt=100, width_cov=100, width_men=100, width_embed=5, family="bernoulli", cov_cancel=False, men_cancel=False)
    history = mcitr.fit(Y, X, A, device="cpu", verbose=1, epochs=50, learning_rate=1e-3)

    D = mcitr.predict(X, A)
    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)
    
    print("---- Unconstrained ----")
    print("---- accuracy: {0} ----".format(accuracy))
    print("---- value: {0} ----".format(value))
    
    cost = np.array([0, 0, 1, 1])
    budgets = 10

    D = mcitr.realign_mckp(X, A, cost = cost, budget=budgets)

    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

    print("---- MCKP ----")
    print("---- accuracy: {0} ----".format(accuracy))
    print("---- value: {0} ----".format(value))
    

    cost_channels = [1, 1]
    budget_channels = [10, 500]

    D = mcitr.realign_random(X, A, cost_channels, budget_channels)

    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

    print("---- Random ----")
    print("---- accuracy: {0} ----".format(accuracy))
    print("---- value: {0} ----".format(value))
    


if __name__ == "__main__":

    demo()