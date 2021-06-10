
from util import getdata
from mcitr import MCITR
import numpy as np
from multiprocessing import Pool
from functools import partial


def run(seed, sample_size, case):

    Y_train, X_train, A_train, optA_train = getdata(sample_size, case=case, seed=seed)
    Y_test, X_test, A_test, optA_test = getdata(2000, case=case, seed=seed + 200)

    mcitr = MCITR(depth_trt=3, depth_cov=3, width_trt=50, width_cov=50, width_embed=3, cov_cancel=False)
    history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, epochs=200)

    D_test = mcitr.predict(X_test, A_test)

    accuracy1, value1 = mcitr.evaluate(Y_test, A_test, D_test, X_test, optA_test)

    mcitr_co = MCITR(depth_trt=3, depth_cov=3, width_trt=50, width_cov=50, width_embed=3, cov_cancel=True) 
    history = mcitr_co.fit(Y_train, X_train, A_train, device="cpu", verbose=0, epochs=200)

    D_test = mcitr_co.predict(X_test, A_test)

    accuracy2, value2 = mcitr_co.evaluate(Y_test, A_test, D_test, X_test, optA_test)

    return accuracy1, value1, accuracy2, value2
    

def main():

    sample_size_list = [200, 400] # training sample size
    case_list = np.arange(5, 9) # data generation cases
    seed_list = np.arange(100) # 200 replicate for each simulation setting

    # unconstrained simulation 

    accuracy_list = np.zeros((2, 4, 100, 2))
    value_list = np.zeros((2, 4, 100, 2))

    for iss, sample_size in enumerate(sample_size_list):

        print("---- sample size: {0} ----".format(sample_size))

        for ic, case in enumerate(case_list):

            print("---- case number: {0} ----".format(case))

            with Pool(8) as p:
                run_part = partial(run, sample_size=sample_size, case=case)
                accuracy1_ls, value1_ls, accuracy2_ls, value2_ls = zip(*p.map(run_part, seed_list))

            accuracy_list[iss, ic, :, 0] = accuracy1_ls
            value_list[iss, ic, :, 0] = value1_ls
            accuracy_list[iss, ic, :, 1] = accuracy2_ls
            value_list[iss, ic, :, 1] = value2_ls

            np.save("accuracy_vs", accuracy_list)
            np.save("value_vs", value_list)


if __name__ == "__main__":
    
    main()

