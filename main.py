

from util import getdata
from mcitr import MCITR
import matplotlib.pyplot as plt
import numpy as np


## demo

def demo():

    Y, X, A, optA = getdata(200, case=1, seed=1)

    mcitr = MCITR(depth_trt=3, depth_cov=3, width_trt=50, width_cov=50, width_embed=3)
    history = mcitr.fit(Y, X, A, device="cpu", verbose=0, epochs=50, learning_rate=5e-2)

    D = mcitr.predict(X, A)
    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)
    
    print("----accuracy: {0}----".format(accuracy))
    print("----value: {0}----".format(value))

    cost = np.array([0, 0, 1, 1])
    budgets = 50

    D = mcitr.realign_mckp(X, A, cost = cost, budget=budgets)

    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

    print("----accuracy: {0}----".format(accuracy))
    print("----value: {0}----".format(value))

    # poor performance
    D = mcitr.realign_mask(X, A, cost = cost, budget=budgets, layer=2, width=100, epochs=1000, lambd=100, learning_rate=1e-1, verbose=0)

    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

    print("----accuracy: {0}----".format(accuracy))
    print("----value: {0}----".format(value))

    cost_channels = [1, 1]
    budget_channels = [50, 200]

    D = mcitr.realign_random(X, A, cost_channels, budget_channels)

    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

    print("----accuracy: {0}----".format(accuracy))
    print("----value: {0}----".format(value))


if __name__ == "__main__":

    demo()

