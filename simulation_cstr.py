# simulation with constraint budget

from src.util import getdata
from src.mcitr import MCITR
from src.tune import tune_mcitr
import numpy as np
from tqdm import tqdm

## sample size: 200
## act_cov: relu, act_men: linear, depth_trt: 5, depth_cov: 5, width: 10, width_embed: 5

# def main():

#     accuracy_array = np.zeros((200, 6))
#     value_array = np.zeros((200, 6))

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(200, case=5, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=5, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="linear", depth_cov=5, depth_trt=5, width_cov=10, width_trt=10, width_embed=5, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         channel_cost = np.array([0, 1, 0])
#         A_test_uq = np.unique(A_test, axis=0)

#         combination_cost = A_test_uq.dot(channel_cost)
#         total_cost = np.sum(optA_test.dot(channel_cost))
#         budgets = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2]) * total_cost

#         for i in range(6):

#             D = mcitr.realign_mckp(X_test, A_test, combination_cost, budgets[i])

#             accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#             accuracy_array[seed, i] = accuracy
#             value_array[seed, i] = value

#     np.savetxt("simulation_2_size_200_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_2_size_200_value.txt", value_array)

## sample size: 400
# act_cov: relu, act_men: relu, depth_trt: 3, depth_cov: 3, width: 256, width_embed: 16

# def main():

#     accuracy_array = np.zeros((200, 6))
#     value_array = np.zeros((200, 6))

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(400, case=5, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=5, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="relu", depth_cov=3, depth_trt=3, width_cov=256, width_trt=256, width_embed=16, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         channel_cost = np.array([0, 1, 0])
#         A_test_uq = np.unique(A_test, axis=0)

#         combination_cost = A_test_uq.dot(channel_cost)
#         total_cost = np.sum(optA_test.dot(channel_cost))
#         budgets = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2]) * total_cost

#         for i in range(6):

#             D = mcitr.realign_mckp(X_test, A_test, combination_cost, budgets[i])

#             accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#             accuracy_array[seed, i] = accuracy
#             value_array[seed, i] = value


#     np.savetxt("simulation_2_size_400_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_2_size_400_value.txt", value_array)

## sample size: 800
# act_cov: relu, act_men: relu, depth_trt: 3, depth_cov: 3, width: 256, width_embed: 16

def main():

    accuracy_array = np.zeros((200, 6))
    value_array = np.zeros((200, 6))

    for seed in tqdm(range(200)):

        Y_train, X_train, A_train, optA_train = getdata(800, case=5, seed=seed)
        Y_test, X_test, A_test, optA_test = getdata(2000, case=5, seed=seed + 200)

        mcitr = MCITR(act_cov="relu", act_men="relu", depth_cov=3, depth_trt=3, width_cov=256, width_trt=256, width_embed=16, cov_cancel=False, men_cancel=False)
        history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

        channel_cost = np.array([0, 1, 0])
        A_test_uq = np.unique(A_test, axis=0)

        combination_cost = A_test_uq.dot(channel_cost)
        total_cost = np.sum(optA_test.dot(channel_cost))
        budgets = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2]) * total_cost

        for i in range(6):

            D = mcitr.realign_mckp(X_test, A_test, combination_cost, budgets[i])

            accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

            accuracy_array[seed, i] = accuracy
            value_array[seed, i] = value


    np.savetxt("simulation_2_size_800_accuracy.txt", accuracy_array)
    np.savetxt("simulation_2_size_800_value.txt", value_array)

if __name__ == "__main__":
    
    main()

