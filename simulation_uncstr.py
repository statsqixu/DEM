# simulation with no cost constraint

from src.util import getdata, getdata3
from src.mcitr import MCITR
from src.tune import tune_mcitr
import numpy as np
from tqdm import tqdm

########### Scenario: 1, sample size: 200, test size: 2000 ############
## tune

# def main():

#     for seed in range(10):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(200, case=1, seed=seed)
#         val_data = getdata(2000, case=1, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [5, 10],
#                 "depth_cov": [1], 
#                 "width": [10, 100, 500, 1000],
#                 "width_embed": [5, 10]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [5, 10],
#                 "depth_cov": [5, 10],
#                 "width": [10, 100, 500, 1000],
#                 "width_embed": [5, 10]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: linear, act_men: linear, depth_trt: 1, depth_cov: 5, width: 500, width_embed: 10

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(200, case=1, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=1, seed=seed + 200)

#         mcitr = MCITR(act_cov="linear", act_men="linear", depth_cov=1, depth_trt=5, width_cov=500, width_trt=500, width_embed=10, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_1_size_200_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_1_size_200_value.txt", value_array)

############ Scenario: 2, sample size: 200, test size: 2000 ###########
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(200, case=2, seed=seed)
#         val_data = getdata(2000, case=2, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [5, 10],
#                 "depth_cov": [1], 
#                 "width": [10, 100, 500, 1000],
#                 "width_embed": [5, 10]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [5, 10],
#                 "depth_cov": [5, 10],
#                 "width": [10, 100, 500, 1000],
#                 "width_embed": [5, 10]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: linear, act_men: relu, depth_trt: 1, depth_cov: 5, width: 100, width_embed: 5

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(200, case=2, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=2, seed=seed + 200)

#         mcitr = MCITR(act_cov="linear", act_men="relu", depth_cov=1, depth_trt=5, width_cov=100, width_trt=100, width_embed=5, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_2_size_200_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_2_size_200_value.txt", value_array)

############ Scenario: 3, sample size: 200, test size: 2000 ###########
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(200, case=3, seed=seed)
#         val_data = getdata(2000, case=3, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: linear, act_men: linear, depth_trt: 3, depth_cov: 1, width: 64, width_embed: 16

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(200, case=3, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=3, seed=seed + 200)

#         mcitr = MCITR(act_cov="linear", act_men="linear", depth_cov=1, depth_trt=3, width_cov=64, width_trt=64, width_embed=16, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_3_size_200_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_3_size_200_value.txt", value_array)


############ Scenario: 4, sample size: 200, test size: 2000 ###########
## tune

# def main():
    
#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(200, case=4, seed=seed)
#         val_data = getdata(2000, case=4, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [5, 10],
#                 "depth_cov": [1], 
#                 "width": [10, 100, 500, 1000],
#                 "width_embed": [5, 10]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [5, 10],
#                 "depth_cov": [5, 10],
#                 "width": [10, 100, 500, 1000],
#                 "width_embed": [5, 10]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: relu, act_men: linear, depth_trt: 10, depth_cov: 10, width: 10, width_embed: 5

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(200, case=4, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=4, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="linear", depth_cov=10, depth_trt=10, width_cov=10, width_trt=10, width_embed=5, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_4_size_200_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_4_size_200_value.txt", value_array)

############ Scenario: 5, sample size: 200, test size: 2000 ###########
## tune

# def main():
    
#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(200, case=5, seed=seed)
#         val_data = getdata(2000, case=5, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [5, 10],
#                 "depth_cov": [1], 
#                 "width": [10, 100, 500, 1000],
#                 "width_embed": [5, 10]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [5, 10],
#                 "depth_cov": [5, 10],
#                 "width": [10, 100, 500, 1000],
#                 "width_embed": [5, 10]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: relu, act_men: linear, depth_trt: 5, depth_cov: 5, width: 10, width_embed: 5

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(200, case=5, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=5, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="linear", depth_cov=5, depth_trt=5, width_cov=10, width_trt=10, width_embed=5, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_5_size_200_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_5_size_200_value.txt", value_array)


############ Scenario: 6, sample size: 200, test size: 2000 ###########
## tune

# def main():
    
#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(200, case=6, seed=seed)
#         val_data = getdata(2000, case=6, seed=seed + 100)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: relu, act_men: linear, depth_trt: 3, depth_cov: 3, width: 256, width_embed: 4

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(200, case=6, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=6, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="linear", depth_cov=3, depth_trt=3, width_cov=256, width_trt=256, width_embed=4, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_6_size_200_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_6_size_200_value.txt", value_array)

########### Scenario: 1, sample size: 400, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(400, case=1, seed=seed)
#         val_data = getdata(2000, case=1, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: linear, act_men: linear, depth_trt: 3, depth_cov: 1, width: 256, width_embed: 8

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(400, case=1, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=1, seed=seed + 200)

#         mcitr = MCITR(act_cov="linear", act_men="linear", depth_cov=1, depth_trt=3, width_cov=256, width_trt=256, width_embed=8, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_1_size_400_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_1_size_400_value.txt", value_array)


########### Scenario: 2, sample size: 400, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(400, case=2, seed=seed)
#         val_data = getdata(2000, case=2, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: linear, act_men: linear, depth_trt: 3, depth_cov: 1, width: 128, width_embed: 4

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(400, case=2, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=2, seed=seed + 200)

#         mcitr = MCITR(act_cov="linear", act_men="linear", depth_cov=1, depth_trt=2, width_cov=128, width_trt=128, width_embed=4, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_2_size_400_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_2_size_400_value.txt", value_array)

########### Scenario: 3, sample size: 400, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(400, case=3, seed=seed)
#         val_data = getdata(2000, case=3, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: linear, act_men: linear, depth_trt: 3, depth_cov: 1, width: 32, width_embed: 8

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(400, case=3, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=3, seed=seed + 200)

#         mcitr = MCITR(act_cov="linear", act_men="linear", depth_cov=1, depth_trt=3, width_cov=32, width_trt=32, width_embed=8, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_3_size_400_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_3_size_400_value.txt", value_array)

########### Scenario: 4, sample size: 400, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(400, case=4, seed=seed)
#         val_data = getdata(2000, case=4, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

# act_cov: relu, act_men: linear, depth_trt: 2, depth_cov: 3, width: 32, width_embed: 8


# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(400, case=4, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=4, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="linear", depth_cov=3, depth_trt=2, width_cov=32, width_trt=32, width_embed=8, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_4_size_400_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_4_size_400_value.txt", value_array)

########### Scenario: 5, sample size: 400, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(400, case=5, seed=seed)
#         val_data = getdata(2000, case=5, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

# act_cov: relu, act_men: relu, depth_trt: 3, depth_cov: 3, width: 256, width_embed: 16


# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(400, case=5, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=5, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="relu", depth_cov=3, depth_trt=3, width_cov=256, width_trt=256, width_embed=16, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_5_size_400_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_5_size_400_value.txt", value_array)

########### Scenario: 6, sample size: 400, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(400, case=6, seed=seed)
#         val_data = getdata(2000, case=6, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

# act_cov: relu, act_men: relu, depth_trt: 3, depth_cov: 3, width: 128, width_embed: 8


# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(400, case=6, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=6, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="relu", depth_cov=3, depth_trt=3, width_cov=128, width_trt=128, width_embed=8, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_6_size_400_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_6_size_400_value.txt", value_array)


########### Scenario: 1, sample size: 800, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(800, case=1, seed=seed)
#         val_data = getdata(2000, case=1, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: linear, act_men: linear, depth_trt: 2, depth_cov: 1, width: 256, width_embed: 4

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(800, case=1, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=1, seed=seed + 200)

#         mcitr = MCITR(act_cov="linear", act_men="linear", depth_cov=1, depth_trt=2, width_cov=256, width_trt=256, width_embed=4, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_1_size_800_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_1_size_800_value.txt", value_array)


########### Scenario: 2, sample size: 400, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(800, case=2, seed=seed)
#         val_data = getdata(2000, case=2, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: linear, act_men: linear, depth_trt: 2, depth_cov: 1, width: 256, width_embed: 8

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(800, case=2, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=2, seed=seed + 200)

#         mcitr = MCITR(act_cov="linear", act_men="linear", depth_cov=1, depth_trt=2, width_cov=256, width_trt=256, width_embed=8, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_2_size_800_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_2_size_800_value.txt", value_array)

########### Scenario: 3, sample size: 800, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(800, case=3, seed=seed)
#         val_data = getdata(2000, case=3, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

## act_cov: linear, act_men: relu, depth_trt: 2, depth_cov: 1, width: 32, width_embed: 16

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(800, case=3, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=3, seed=seed + 200)

#         mcitr = MCITR(act_cov="linear", act_men="relu", depth_cov=1, depth_trt=2, width_cov=32, width_trt=32, width_embed=16, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_3_size_800_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_3_size_800_value.txt", value_array)

########### Scenario: 4, sample size: 800, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(800, case=4, seed=seed)
#         val_data = getdata(2000, case=4, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

# act_cov: relu, act_men: linear, depth_trt: 2, depth_cov: 3, width: 32, width_embed: 16


# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(800, case=4, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=4, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="linear", depth_cov=3, depth_trt=2, width_cov=32, width_trt=32, width_embed=16, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_4_size_800_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_4_size_800_value.txt", value_array)

########### Scenario: 6, sample size: 800, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(800, case=5, seed=seed)
#         val_data = getdata(2000, case=5, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

# act_cov: relu, act_men: relu, depth_trt: 3, depth_cov: 3, width: 256, width_embed: 16


# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(800, case=5, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=5, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="relu", depth_cov=3, depth_trt=3, width_cov=256, width_trt=256, width_embed=16, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_5_size_800_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_5_size_800_value.txt", value_array)

########### Scenario: 6, sample size: 800, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata(800, case=6, seed=seed)
#         val_data = getdata(2000, case=6, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

# act_cov: relu, act_men: relu, depth_trt: 3, depth_cov: 3, width: 128, width_embed: 8


# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata(800, case=6, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata(2000, case=6, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="linear", depth_cov=2, depth_trt=2, width_cov=128, width_trt=128, width_embed=8, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_6_size_800_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_6_size_800_value.txt", value_array)

########### Scenario: 7, sample size: 200, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata3(200, case=1, seed=seed)
#         val_data = getdata3(2000, case=1, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

# act_cov: relu, act_men: linear, depth_trt: 2, depth_cov: 3, width: 256, width_embed: 8


# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata3(200, case=1, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata3(2000, case=1, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="linear", depth_cov=3, depth_trt=2, width_cov=256, width_trt=256, width_embed=8, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_7_size_200_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_7_size_200_value.txt", value_array)


########### Scenario: 7, sample size: 400, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata3(400, case=1, seed=seed)
#         val_data = getdata3(2000, case=1, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

# act_cov: relu, act_men: relu, depth_trt: 3, depth_cov: 3, width: 32, width_embed: 8


# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata3(400, case=1, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata3(2000, case=1, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="linear", depth_cov=3, depth_trt=2, width_cov=256, width_trt=256, width_embed=8, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_7_size_400_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_7_size_400_value.txt", value_array)

########### Scenario: 7, sample size: 800, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata3(800, case=1, seed=seed)
#         val_data = getdata3(2000, case=1, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

# act_cov: relu, act_men: relu, depth_trt: 2, depth_cov: 2, width: 128, width_embed: 4

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata3(800, case=1, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata3(2000, case=1, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="linear", depth_cov=3, depth_trt=2, width_cov=256, width_trt=256, width_embed=8, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_7_size_800_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_7_size_800_value.txt", value_array)

########### Scenario: 8, sample size: 200, test size: 2000 ############
## tune

# def main():

#     for seed in range(5):

#         print("Start tune parameters for data set {0}".format(seed + 1))
#         train_data = getdata3(200, case=2, seed=seed)
#         val_data = getdata3(2000, case=2, seed=seed + 200)

#         params = [
#             {
#                 "act_cov": ["linear"], 
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [1], 
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             },
#             {
#                 "act_cov": ["relu"],
#                 "act_men": ["linear", "relu"],
#                 "depth_trt": [2, 3],
#                 "depth_cov": [2, 3],
#                 "width": [32, 64, 128, 256],
#                 "width_embed": [4, 8, 16]
#             }
#         ]

#         tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

# act_cov: relu, act_men: relu, depth_trt: 3, depth_cov: 3, width: 256, width_embed: 16

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata3(200, case=2, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata3(2000, case=2, seed=seed + 200)

#         mcitr = MCITR(act_cov="relu", act_men="relu", depth_cov=3, depth_trt=3, width_cov=256, width_trt=256, width_embed=16, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_8_size_200_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_8_size_200_value.txt", value_array)

########### Scenario: 8, sample size: 400, test size: 2000 ############

# def main():
    
    # accuracy_list = []
    # value_list = []

    # for seed in tqdm(range(200)):

    #     Y_train, X_train, A_train, optA_train = getdata3(400, case=2, seed=seed)
    #     Y_test, X_test, A_test, optA_test = getdata3(2000, case=2, seed=seed + 200)

    #     mcitr = MCITR(act_cov="relu", act_men="relu", depth_cov=3, depth_trt=3, width_cov=256, width_trt=256, width_embed=16, cov_cancel=False, men_cancel=False)
    #     history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

    #     D = mcitr.predict(X_test, A_test)

    #     accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

    #     accuracy_list.append(accuracy)
    #     value_list.append(value)

    # accuracy_array = np.array(accuracy_list)
    # value_array = np.array(value_list)

    # np.savetxt("simulation_1_scenario_8_size_400_accuracy.txt", accuracy_array)
    # np.savetxt("simulation_1_scenario_8_size_400_value.txt", value_array)

########### Scenario: 8, sample size: 800, test size: 2000 ############

# def main():
    
    # accuracy_list = []
    # value_list = []

    # for seed in tqdm(range(200)):

    #     Y_train, X_train, A_train, optA_train = getdata3(800, case=2, seed=seed)
    #     Y_test, X_test, A_test, optA_test = getdata3(2000, case=2, seed=seed + 200)

    #     mcitr = MCITR(act_cov="relu", act_men="relu", depth_cov=3, depth_trt=3, width_cov=256, width_trt=256, width_embed=16, cov_cancel=False, men_cancel=False)
    #     history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

    #     D = mcitr.predict(X_test, A_test)

    #     accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

    #     accuracy_list.append(accuracy)
    #     value_list.append(value)

    # accuracy_array = np.array(accuracy_list)
    # value_array = np.array(value_list)

    # np.savetxt("simulation_1_scenario_8_size_800_accuracy.txt", accuracy_array)
    # np.savetxt("simulation_1_scenario_8_size_800_value.txt", value_array)

########### Scenario: 9, sample size: 200, test size: 2000 ############
## tune

# def main():

    # for seed in range(5):

    #     print("Start tune parameters for data set {0}".format(seed + 1))
    #     train_data = getdata3(200, case=3, seed=seed)
    #     val_data = getdata3(2000, case=3, seed=seed + 200)

    #     params = [
    #         {
    #             "act_cov": ["linear"], 
    #             "act_men": ["linear", "relu"],
    #             "depth_trt": [2, 3],
    #             "depth_cov": [1], 
    #             "width": [32, 64, 128, 256],
    #             "width_embed": [4, 8, 16]
    #         },
    #         {
    #             "act_cov": ["relu"],
    #             "act_men": ["linear", "relu"],
    #             "depth_trt": [2, 3],
    #             "depth_cov": [2, 3],
    #             "width": [32, 64, 128, 256],
    #             "width_embed": [4, 8, 16]
    #         }
    #     ]

    #     tune_result = tune_mcitr(params, train_data, val_data, save=True, save_path="tune_result_{0}.csv".format(seed))

# act_cov: linear, act_men: linear, depth_trt: 2, depth_cov: 1, width: 32, width_embed: 8

# def main():

#     accuracy_list = []
#     value_list = []

#     for seed in tqdm(range(200)):

#         Y_train, X_train, A_train, optA_train = getdata3(200, case=3, seed=seed)
#         Y_test, X_test, A_test, optA_test = getdata3(2000, case=3, seed=seed + 200)

#         mcitr = MCITR(act_cov="linear", act_men="linear", depth_cov=1, depth_trt=2, width_cov=32, width_trt=32, width_embed=8, cov_cancel=False, men_cancel=False)
#         history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

#         D = mcitr.predict(X_test, A_test)

#         accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

#         accuracy_list.append(accuracy)
#         value_list.append(value)

#     accuracy_array = np.array(accuracy_list)
#     value_array = np.array(value_list)

#     np.savetxt("simulation_1_scenario_9_size_200_accuracy.txt", accuracy_array)
#     np.savetxt("simulation_1_scenario_9_size_200_value.txt", value_array)

########### Scenario: 9, sample size: 400, test size: 2000 ############

# def main():
    
    # accuracy_list = []
    # value_list = []

    # for seed in tqdm(range(200)):

    #     Y_train, X_train, A_train, optA_train = getdata3(400, case=3, seed=seed)
    #     Y_test, X_test, A_test, optA_test = getdata3(2000, case=3, seed=seed + 200)

    #     mcitr = MCITR(act_cov="linear", act_men="linear", depth_cov=1, depth_trt=2, width_cov=32, width_trt=32, width_embed=8, cov_cancel=False, men_cancel=False)
    #     history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

    #     D = mcitr.predict(X_test, A_test)

    #     accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

    #     accuracy_list.append(accuracy)
    #     value_list.append(value)

    # accuracy_array = np.array(accuracy_list)
    # value_array = np.array(value_list)

    # np.savetxt("simulation_1_scenario_9_size_400_accuracy.txt", accuracy_array)
    # np.savetxt("simulation_1_scenario_9_size_400_value.txt", value_array)

########### Scenario: 9, sample size: 800, test size: 2000 ############

# def main():
    
    # accuracy_list = []
    # value_list = []

    # for seed in tqdm(range(200)):

    #     Y_train, X_train, A_train, optA_train = getdata3(800, case=3, seed=seed)
    #     Y_test, X_test, A_test, optA_test = getdata3(2000, case=3, seed=seed + 200)

    #     mcitr = MCITR(act_cov="linear", act_men="linear", depth_cov=1, depth_trt=2, width_cov=32, width_trt=32, width_embed=8, cov_cancel=False, men_cancel=False)
    #     history = mcitr.fit(Y_train, X_train, A_train, device="cpu", verbose=0, learning_rate=1e-2)

    #     D = mcitr.predict(X_test, A_test)

    #     accuracy, value = mcitr.evaluate(Y_test, A_test, D, X_test, optA=optA_test)

    #     accuracy_list.append(accuracy)
    #     value_list.append(value)

    # accuracy_array = np.array(accuracy_list)
    # value_array = np.array(value_list)

    # np.savetxt("simulation_1_scenario_9_size_800_accuracy.txt", accuracy_array)
    # np.savetxt("simulation_1_scenario_9_size_800_value.txt", value_array)


if __name__ == "__main__":
    
    main()

