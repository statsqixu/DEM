

from util import getdata, getdata2, getdata3
from mcitr import MCITR
import numpy as np


## demo

def demo():

    Y, X, A, optA = getdata3(500, case=1, seed=1, family="gaussian")

    mcitr = MCITR(depth_trt=3, depth_cov=3, depth_men=3, width_trt=1000, width_cov=1000, width_men=100, width_embed=5, family="gaussian", cov_cancel=False, men_cancel=False)
    history = mcitr.fit(Y, X, A, device="cpu", verbose=1, epochs=100, learning_rate=1e-2)

    D = mcitr.predict(X, A)
    accuracy, value = mcitr.evaluate(Y, A, D, X, optA)
    
    print("---- Unconstrained ----")
    print("---- accuracy: {0} ----".format(accuracy))
    print("---- value: {0} ----".format(value))
    


if __name__ == "__main__":

    demo()