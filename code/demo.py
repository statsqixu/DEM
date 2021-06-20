

from util import getdata, getdata2, getdata3
from mcitr import MCITR
import numpy as np


## demo

def demo():

    Y_train, X_train, A_train, optA_train = getdata3(500, case=2, seed=1, family="gaussian")
    Y_test, X_test, A_test, optA_test = getdata3(2000, case=2, seed=1, family="gaussian")

    mcitr = MCITR(depth_trt=3, depth_cov=3, depth_men=3, width_trt=1000, width_cov=1000, width_men=100, width_embed=5, family="gaussian", cov_cancel=False, men_cancel=False)
    history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=1, epochs=100, learning_rate=5e-3)

    D = mcitr.predict(X_test, A_test)
    accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA_test)
    
    print("---- Unconstrained ----")
    print("---- accuracy: {0} ----".format(accuracy))
    print("---- value: {0} ----".format(value))    

if __name__ == "__main__":

    demo()