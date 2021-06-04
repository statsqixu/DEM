
from util import getdata
from mcitr import MCITR
import numpy as np
from tqdm import tqdm


def run(seed):

    global accuarcy_list, value_list

    Y_train, X_train, A_train, optA_train = getdata(sample_size, case=case, seed=seed)
    Y_test, X_test, A_test, optA_test = getdata(2000, case=case, seed=seed + 200)

    mcitr = MCITR()
    history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, epochs=200)

    D_test = mcitr.predict(X_test, A_test)

    accuracy, value = mcitr.evaluate(Y_test, A_test, D_test, X_test, optA_test)

    accuracy_list[iss, ic, seed] = accuracy
    value_list[iss, ic, seed] = value


def main():

    sample_size_list = [200, 400, 800, 2000] # training sample size
    case_list = np.arange(1, 9) # data generation cases
    seed_list = np.arange(200) # 200 replicate for each simulation setting

    # unconstrained simulation 

    accuracy_list = np.zeros((4, 8, 200))
    value_list = np.zeros((4, 8, 200))

    for iss, sample_size in enumerate(sample_size_list):

        print("---- sample size: {0} ----".format(sample_size))

        for ic, case in enumerate(case_list):

            print("---- case number: {0} ----".format(case))

            with Pool(8) as p:

                p.map(run, seed_list)


            np.save("accuracy_uncstr", accuracy_list)
            np.save("value_uncstr", value_list)


if __name__ == "__main__":
    
    main()

