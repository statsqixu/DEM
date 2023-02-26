
from src.util import *
from src.dem import *


## demo

X_train, A_train, Y_train, optA_train = generate_data(500, case=4, seed=0, mode="ps")
X_test, A_test, Y_test, optA_test = generate_data(2000, case=4, seed=200, mode="ps")

itr = ITR(embed_dim = 2, trt_encoder = "nn",
                 cov_encoder = "nn", cov_act="relu",
                 cov_layer=3, cov_width=32)

history = itr.fit(X_train, A_train, Y_train, mode="ps")

D = itr.predict(X_test, A_test)

output = itr.evaluate(Y_test, A_test, D, optA_test, X_test)

#print("---- Unconstrained ----")
print("---- accuracy: {0} ----".format(output[0]))
print("---- value: {0} ----".format(output[1]))