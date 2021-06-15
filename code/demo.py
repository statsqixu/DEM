

from util import getdata
from mcitr import MCITR
import numpy as np


## demo

def demo():

    Y, X, A, optA = getdata(500, case=1, seed=1)

    mcitr = MCITR(depth_trt=3, depth_cov=3, width_trt=50, width_cov=50, width_embed=3, cov_cancel=True)
    history = mcitr.fit(Y, X, A, device="cpu", verbose=0, epochs=50, learning_rate=5e-2)

    D = mcitr.predict(X, A)
    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)
    
    print("---- Unconstrained ----")
    print("----accuracy: {0}----".format(accuracy))
    print("----value: {0}----".format(value))
    
    cost = np.array([0, 0, 1, 1])
    budgets = 20

    D = mcitr.realign_mckp(X, A, cost = cost, budget=budgets)

    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

    print("---- MCKP ----")
    print("----accuracy: {0}----".format(accuracy))
    print("----value: {0}----".format(value))
    

    cost_channels = [1, 1]
    budget_channels = [20, 500]

    D = mcitr.realign_random(X, A, cost_channels, budget_channels)

    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

    print("---- Random ----")
    print("----accuracy: {0}----".format(accuracy))
    print("----value: {0}----".format(value))
    


if __name__ == "__main__":

    demo()

