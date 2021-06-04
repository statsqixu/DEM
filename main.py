

from util import getdata
from mcitr import MCITR
import matplotlib.pyplot as plt
import numpy as np


## demo

def demo():

    # Y, X, A, optA = getdata(200, case=2, seed=2)

    # mcitr = MCITR(depth_trt=3, depth_cov=3, width_trt=50, width_cov=50, width_embed=3)
    # history = mcitr.fit(Y, X, A, device="cpu", verbose=1, epochs=200, learning_rate=5e-2)

    # D = mcitr.predict(X, A)
    # accuracy, value = mcitr.evaluate(Y, A, D, X, optA)
    
    # print("----accuracy: {0}----".format(accuracy))
    # print("----value: {0}----".format(value))

    # cost = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # budgets = 20

    # D = mcitr.realign_mckp(X, A, cost = cost, budget=budgets)

    # accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

    # print("----accuracy: {0}----".format(accuracy))
    # print("----value: {0}----".format(value))

    Y, X, A, optA = getdata(200, case=1)

    plt.hist(Y)
    plt.savefig("python_outcome_hist.jpg", dpi=300)

if __name__ == "__main__":

    demo()

