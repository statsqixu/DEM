
from util import getdata
from mcitr import MCITR
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def run(seed):

    global accuracy_list, value_list

    Y_train, X_train, A_train, optA_train = getdata(sample_size, case=case, seed=seed)
    Y_test, X_test, A_test, optA_test = getdata(2000, case=case, seed=seed + 200)

    mcitr = MCITR()
    history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, epochs=200)

    D_test = mcitr.predict(X_test, A_test)

    accuracy, value = mcitr.evaluate(Y_test, A_test, D_test, X_test, optA_test)

    accuracy_list[iss, ic, seed, iq, 0] = accuracy
    value_list[iss, ic, seed, iq, 0] = value

    D_test_mckp = mcitr.realign_mckp(X_test, A_test, cost_combinations, budget_combinations)

    accuracy, value = mcitr.evaluate(Y_test, A_test, D_test_mckp, X_test, optA_test)

    accuracy_list[iss, ic, seed, iq, 1] = accuracy
    value_list[iss, ic, seed, iq, 1] = value

    D_test_rdm = mcitr.realign_random(X_test, A_test, cost_channels, budget_channels)

    accuracy, value = mcitr.evaluate(Y_test, A_test, D_test_rdm, X_test, optA_test)

    accuracy_list[iss, ic, seed, iq, 2] = accuracy
    value_list[iss, ic, seed, iq, 2] = value


def main():

    # 2-channel simulation 

    sample_size_list = [200, 400, 800, 2000] # training sample size
    case_list = [1, 2, 3, 4, 5, 6, 7, 8] # data generation cases
    seed_list = np.arange(200) # 200 replicate for each simulation setting
    quantile_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    accuracy_list = np.zeros((4, 8, 200, 5, 3))
    value_list = np.zeros((4, 8, 200, 5, 3))

    
    for iss, sample_size in enumerate(sample_size_list):

        print("---- sample size: {0} ----".format(sample_size))

        for ic, case in enumerate(case_list):

            print("---- case number: {0} ----".format(case))

            for iq, quantile in enumerate(quantile_list):

                print("---- quantile: {0} ----".format(quantile))

                if case in [1, 3, 5, 7]:

                    cost_channels = [1, 1]
                    cost_combinations = [0, 0, 1, 1]

                    budget_channels = [2000 * quantile, 2000 * 2]
                    budget_combinations = 2000 * quantile

                elif case in [2, 4, 6, 8]:
                    
                    cost_channels = [1, 1, 1]
                    cost_combinations = [0, 0, 0, 0, 1, 1, 1, 1]

                    budget_channels = [2000 * quantile, 2000, 2000 * 2]
                    budget_combinations = 2000 * quantile

                with Pool(8) as p:

                    p.map(run, seed_list)

                np.save("accuracy_cstr", accuracy_list)
                np.save("value_cstr", value_list)



if __name__ == "__main__":
    
    main()

