from tabnanny import verbose
from util import getdata
from mcitr import MCITR
import numpy as np


## demo

def demo():

    Y, X, A, optA = getdata(200, case=1, seed=1)

    mcitr = MCITR(width_embed=2)
    history = mcitr.fit(Y, X, A, device="cpu", verbose=0, epochs=200)

    D = mcitr.predict(X, A)
    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)
    
    print("----accuracy: {0}----".format(accuracy))
    print("----value: {0}----".format(value))

    cost = np.array([0, 1, 0, 1])
    budgets = 50

    D = mcitr.realign_mckp(X, A, cost = cost, budget=budgets)

    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)

    print("----accuracy: {0}----".format(accuracy))
    print("----value: {0}----".format(value))

if __name__ == "__main__":

    demo()

