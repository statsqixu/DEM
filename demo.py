

from src.util import getdata
from src.mcitr import MCITR
import numpy as np


## demo

def demo():

    Y_train, X_train, A_train, optA_train = getdata(200, case=6, seed=1, family="gaussian")
    Y_test, X_test, A_test, optA_test = getdata(2000, case=6, seed=201, family="gaussian")

    mcitr = MCITR(depth_trt=3, depth_cov=3, depth_men=1, width_trt=128, width_cov=128, width_men=8, width_embed=16, family="gaussian", cov_cancel=False, men_cancel=False)
    history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, epochs=100, learning_rate=5e-3)

    D = mcitr.predict(X_test, A_test)
    accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA_test)
    
    print("---- Unconstrained ----")
    print("---- accuracy: {0} ----".format(accuracy))
    print("---- value: {0} ----".format(value))    

if __name__ == "__main__":

    demo()