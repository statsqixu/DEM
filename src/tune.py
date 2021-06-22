from mcitr import MCITR
from util import getdata
import pandas as pd
from sklearn.model_selection import ParameterGrid


def tune_mcitr(params, train_data, val_data, save=False, save_path=None, verbose=0):

    grid = ParameterGrid(params)

    Y_train, X_train, A_train, optA_train = train_data
    Y_val, X_val, A_val, optA_val = val_data

    grid_result = []

    for config in grid:

        mcitr = MCITR(act_cov=config["act_cov"],
                        act_men=config["act_men"],
                        depth_trt=config["depth_trt"],
                        depth_cov=config["depth_cov"],
                        width_trt=config["width"],
                        width_cov=config["width"],
                        width_embed=config["width_embed"],
                        cov_cancel=False, men_cancel=False)
        history = mcitr.fit(Y_train, X_train, A_train, device="cpu",
                            verbose=0, learning_rate=1e-2)

        D = mcitr.predict(X_val, A_val)

        accuracy, value = mcitr.evaluate(Y_val, A_val, D, X_val, optA_val)

        config["accuracy"] = accuracy
        config["value"] = value

        if verbose == 1:
            print(config)

        grid_result.append(config)

    output = pd.DataFrame.from_dict(grid_result)

    if save:
        output.to_csv(save_path)

    return output

