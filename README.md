# IOTR

## Python implementation of Individualized Omnichannel Treatment Rule

- Package Requirement
```
numpy		version 1.19.2
torch		version 1.7.1+cu101
sklearn		version 0.0
matplotlib	version 3.4.1
```

- Demo
```
from src.util import getdata
from src.mcitr import MCITR
import numpy as np


## demo

def demo():

    Y_train, X_train, A_train, optA_train = getdata(200, case=1, seed=0, family="gaussian")
    Y_test, X_test, A_test, optA_test = getdata(2000, case=1, seed=200, family="gaussian")

    mcitr = MCITR(act_trt="relu", act_cov="relu", depth_trt=5, depth_cov=5, width_trt=256, width_cov=256, width_embed=8)
    history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=1, epochs=500, learning_rate=0.001)

    D = mcitr.predict(X_test, A_test)
    accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA_test)
    
    print("---- Unconstrained ----")
    print("---- accuracy: {0} ----".format(accuracy))
    print("---- value: {0} ----".format(value))    

if __name__ == "__main__":

    demo()
```


