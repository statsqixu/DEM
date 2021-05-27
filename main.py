from util import getdata
from mcitr import MCITR
import numpy as np


## demo

def demo():

    Y, X, A, optA = getdata(20, case=1)

    mcitr = MCITR(width_embed=2)
    history = mcitr.fit(Y, X, A, device="cpu", verbose=0, epochs=200)

    D = mcitr.predict(X, A)
    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)
    
    print("----accuracy: {0}----".format(accuracy))
    print("----value: {0}----".format(value))

    cost = np.array([0, 10, 20, 30])

    budgets = np.random.uniform(low=25, high=40, size=(20, ))

    D1, comp = mcitr.realign(X, A, cost = cost, budgets=budgets, lambda_1=0.05, budget_level="individual")
    print("----Compliance: {0}----".format(np.sum(comp) / 20))

    accuracy, value = mcitr.evaluate(Y, A, D1, X, optA)

    print("----accuracy: {0}----".format(accuracy))
    print("----value: {0}----".format(value))



if __name__ == "__main__":

    demo()

