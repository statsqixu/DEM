from src.dem import *
from src.util import *
import pandas as pd
import numpy as np
import torch
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

import os

def train(config):

    """
    Train the Double Encoder Model given the configuration of hyperparameters.
    """

    X_train, A_train, Y_train, optA_train = train_data # need to be specified
    X_val, A_val, Y_val, optA_val = valid_data # need to be specified
    
    accs = []
    vals = []


    for rnd in range(10): # set the random seed for each trial
        
        torch.manual_seed(rnd)

        if (config["trt_encoder"] == "nn") and (config["cov_encoder"] == "nn"):

            model = ITR(embed_dim=int(config["embed_dim"]), trt_encoder=config["trt_encoder"],
                        trt_layer=int(config["trt_layer"]), trt_width=int(config["trt_width"]),
                        trt_act=config["trt_act"], cov_encoder=config["cov_encoder"],
                        cov_layer=int(config["cov_layer"]), cov_width=int(config["cov_width"]),
                        cov_act=config["cov_act"])
            
        elif (config["trt_encoder"] == "nn") and (config["cov_encoder"] == "poly"):

            model = ITR(embed_dim=int(config["embed_dim"]), trt_encoder=config["trt_encoder"],
                        trt_layer=int(config["trt_layer"]), trt_width=int(config["trt_width"]),
                        trt_act=config["trt_act"], cov_encoder=config["cov_encoder"],
                        cov_degree=int(config["cov_degree"]))

        elif (config["trt_encoder"] == "nn") and (config["cov_encoder"] == "bs"):

            model = ITR(embed_dim=int(config["embed_dim"]), trt_encoder=config["trt_encoder"],
                        trt_layer=int(config["trt_layer"]), trt_width=int(config["trt_width"]),
                        trt_act=config["trt_act"], cov_encoder=config["cov_encoder"],
                        cov_bs_order=int(config["cov_bs_order"]), cov_bs_bases=int(config["cov_bs_bases"]))
            
        elif (config["trt_encoder"] == "dict") and (config["cov_encoder"] == "nn"):

            model = ITR(embed_dim=int(config["embed_dim"]), trt_encoder=config["trt_encoder"],
                        trt_num=int(config["trt_num"]), cov_encoder=config["cov_encoder"],
                        cov_layer=int(config["cov_layer"]), cov_width=int(config["cov_width"]),
                        cov_act=config["cov_act"]) 

        elif (config["trt_encoder"] == "dict") and (config["cov_encoder"] == "poly"):

            model = ITR(embed_dim=int(config["embed_dim"]), trt_encoder=config["trt_encoder"],
                        trt_num=int(config["trt_num"]), cov_encoder=config["cov_encoder"],
                        cov_degree=int(config["cov_degree"]))
            
        elif (config["trt_encoder"] == "dict") and (config["cov_encoder"] == "bs"):

            model = ITR(embed_dim=int(config["embed_dim"]), trt_encoder=config["trt_encoder"],
                        trt_num=int(config["trt_num"]), cov_encoder=config["cov_encoder"],
                        cov_bs_order=int(config["cov_bs_order"]), cov_bs_bases=int(config["cov_bs_bases"]))

        _ = model.fit(X_train, A_train, Y_train, device="cpu", mode=config["mode"],
                            epochs=config["epochs"], batch_size=config["batch_size"],
                            learning_rate=config["lr"], 
                            weight_decay=config["weight_decay"],
                            interactive_weight_decay=config["interactive_weight_decay"],
                            trt_free_model=config["trt_free_model"],
                            ps_model=config["ps_model"])

        D = model.predict(X_val, A_val)
        output = model.evaluate(Y_val, A_val, D, optA_val, X_val)

        if optA_val is not None:

            accs.append(output[0])
            vals.append(output[1])

            tune.report({"accuracy_mean": np.mean(accs),
                 "accuracy_std": np.std(accs),
                 "value_mean": np.mean(vals),
                 "value_std": np.std(vals)})

        else:

            vals.append(output[0])

            tune.report({"value_mean": np.mean(vals),
                 "value_std": np.std(vals)}) 

    torch.save(model.model.state_dict(), "./model.pth")

# default search space of hyperparameters

# search_space = {
#     "mode": tune.grid_search(["randomized", "ps"]),
#     "trt_encoder": tune.grid_search(["nn", "dict"]),
#     "embed_dim": tune.choice([2, 3, 4]),
#     "trt_layer": tune.randint(3, 8),
#     "trt_width": tune.choice([16, 32, 64, 128]),
#     "trt_act": tune.choice(["relu", "linear"]),
#     "cov_encoder": tune.grid_search(["nn", "poly", "bs"]),
#     "cov_layer": tune.randint(3, 8),
#     "cov_width": tune.choice([16, 32, 64, 128]),
#     "cov_act": tune.choice(["relu", "linear"]),
#     "cov_degree": tune.choice([2, 3, 4]),
#     "cov_bs_order": tune.choice([2, 3, 4]),
#     "cov_bs_bases": tune.choice([2, 3, 4]),
#     "epochs": tune.choice([20, 50, 100]),
#     "batch_size": tune.choice([8, 16, 32]),
#     "lr": tune.loguniform(1e-4, 1e-1),
#     "weight_decay": tune.loguniform(1e-4, 1e-1),
#     "interactive_weight_decay": tune.loguniform(1e-4, 1e-1),
#     "trt_free_model": tune.grid_search(["nn", "linear"]),
#     "ps_model": tune.grid_search(["multinomial", "nn"]),
# }

def tuner(search_space, results_dir):

    tuner = tune.Tuner(
        train,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=BasicVariantGenerator(random_state=0), # for reproducibility
            num_samples=50,
        ),
    )

    results = tune.fit()
    results_df = results.get_dataframe()

    results_df.to_csv(results_dir + "/tune_results.csv", index=False)
