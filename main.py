from util import getdata, plot_train_history
from mcitr import MCITR


## demo

def demo():

    Y, X, A, optA = getdata(1000, case=1)

    mcitr = MCITR()
    history = mcitr.fit(Y, X, A)
    D = mcitr.predict(X, A)
    accuracy, value = mcitr.evaluate(Y, A, D, optA=optA)

    print("The value function for estimated ITR is {0}".format(value))
    print("The accuracy for estimated ITR is {0}".format(accuracy))


if __name__ == "__main__":

    demo()