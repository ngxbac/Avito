import json
import pandas as pd
import bcolz
import numpy as np
import os

import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

import models
import datasets as d
import utils
import json


def main():
    # Load json config
    config = json.load(open("config.json"))

    # Load data and token len of embedding layers
    X = np.load("X_train.npy")
    y = np.load("y_train.npy")
    test = np.load("X_test.npy")
    token_len = np.load("token_len.npy")

    # Create train/valid dataset and dataloader
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=1700)

    # Leave most parameters as default
    param = {
        'objective': 'reg:logistic',  # Specify multiclass classification
        'tree_method': 'gpu_hist',  # Use GPU accelerated algorithm
        'bosster': 'gdtree',
        'eval_metric': 'rmse',
        'eta': 0.05,
        'max_depth': 13,
        'min_child_weight': 2,
        'gamma': 0,
        'subsample': 0.7,
        'alpha': 0,
        'lamda': 0
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(test)


    # Specify sufficient boosting iterations to reach a minimum
    num_round = 1500
    gpu_res = {}  # Store accuracy result
    # Train model
    bst = xgb.train(param, dtrain, num_round, evals=[(dtrain, 'train'), (dval, 'test')],
                    evals_result=gpu_res,
                    verbose_eval=10, early_stopping_rounds=50)

    deal_probability = bst.predict(dtest)
    submission = pd.read_csv(config["sample_submission"])
    submission['deal_probability'] = deal_probability
    submission.to_csv(f"submission_xgboost.csv", index=False)

if __name__ == '__main__':
    main()