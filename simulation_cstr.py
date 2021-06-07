
from util import getdata
from mcitr import MCITR
import numpy as np
from multiprocessing import Pool
from functools import partial


def run(seed, sample_size, case, cost_combinations, budget_combinations, cost_channels, budget_channels):

    Y_train, X_train, A_train, optA_train = getdata(sample_size, case=case, seed=seed)
    Y_test, X_test, A_test, optA_test = getdata(2000, case=case, seed=seed + 200)

    mcitr = MCITR()
    history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, epochs=200)

    D_test = mcitr.predict(X_test, A_test)

    accuracy1, value1 = mcitr.evaluate(Y_test, A_test, D_test, X_test, optA_test)

    D_test_mckp = mcitr.realign_mckp(X_test, A_test, cost_combinations, budget_combinations)

    accuracy2, value2 = mcitr.evaluate(Y_test, A_test, D_test_mckp, X_test, optA_test)

    D_test_mask = mcitr.realign_mask(X_test, A_test, cost_combinations, budget_combinations)

    accuracy3, value3 = mcitr.evaluate(Y_test, A_test, D_test_mask, X_test, optA_test)

    D_test_rdm = mcitr.realign_random(X_test, A_test, cost_channels, budget_channels)

    accuracy4, value4 = mcitr.evaluate(Y_test, A_test, D_test_rdm, X_test, optA_test)

    return accuracy1, value1, accuracy2, value2, accuracy3, value3, accuracy4, value4


def main():

    # 2-channel simulation 

    sample_size_list = [200, 400, 800, 2000] # training sample size
    case_list = [1, 2, 3, 4] # data generation cases
    seed_list = np.arange(200) # 200 replicate for each simulation setting
    quantile_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    accuracy_list = np.zeros((4, 4, 200, 5, 4))
    value_list = np.zeros((4, 4, 200, 5, 4))

    
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

                    budget_channels = [2000 * quantile, 2000 * 2, 2000 * 2]
                    budget_combinations = 2000 * quantile

                with Pool(8) as p:
                    run_part = partial(run, sample_size=sample_size, case=case, 
                                        cost_combinations=cost_combinations,
                                        budget_combinations=budget_combinations,
                                        cost_channels=cost_channels,
                                        budget_channels=budget_channels)
                    accuracy1_list, value1_list, accuracy2_list, value2_list, accuracy3_list, value3_list, accuracy4_list, value4_list = zip(*p.map(run_part, seed_list))

                    accuracy_list[iss, ic, :, iq, 0] = accuracy1_list
                    accuracy_list[iss, ic, :, iq, 1] = accuracy2_list
                    accuracy_list[iss, ic, :, iq, 2] = accuracy3_list
                    accuracy_list[iss, ic, :, iq, 3] = accuracy4_list

                    value_list[iss, ic, :, iq, 0] = value1_list
                    value_list[iss, ic, :, iq, 1] = value2_list
                    value_list[iss, ic, :, iq, 2] = value3_list
                    value_list[iss, ic, :, iq, 3] = value4_list

                np.save("accuracy_cstr", accuracy_list)
                np.save("value_cstr", value_list)



if __name__ == "__main__":
    
    main()

