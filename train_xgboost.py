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

from scipy.sparse import hstack


def main():
    # Load json config
    config = json.load(open("config.json"))
    extracted_features_root = config["extracted_features"]
    # Load data and token len of embedding layers
    print("[+] Load features ...")
    y = utils.load_features(extracted_features_root, "y_train")
    token_len = utils.load_features(extracted_features_root, "token_len")

    X_train_num = utils.load_features(extracted_features_root, "X_train_num")
    X_train_cat = utils.load_features(extracted_features_root, "X_train_cat")
    X_train_desc = utils.load_features(extracted_features_root, "X_train_desc").any()
    X_train_title = utils.load_features(extracted_features_root, "X_train_title").any()
    X_train_param = utils.load_features(extracted_features_root, "X_train_param").any()

    X_test_num = utils.load_features(extracted_features_root, "X_test_num")
    X_test_cat = utils.load_features(extracted_features_root, "X_test_cat")
    X_test_desc = utils.load_features(extracted_features_root, "X_test_desc").any()
    X_test_title = utils.load_features(extracted_features_root, "X_test_title").any()
    X_test_param = utils.load_features(extracted_features_root, "X_test_param").any()


    # X = np.concatenate((X_train_num, X_train_cat, X_train_desc, X_train_title), axis=1)
    # X = np.concatenate((X_train_num, X_train_cat, X_pred_train), axis=1)
    X = hstack([X_train_num, X_train_cat, X_train_desc, X_train_title])
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

    # test = np.concatenate((X_test_num, X_test_cat), axis=1)
    # test = np.concatenate((X_test_num, X_test_cat, X_test_desc, X_test_title), axis=1)
    test = hstack([X_test_num, X_test_cat, X_test_desc, X_test_title])

    # Leave most parameters as default
    param = {
        'objective': 'reg:logistic',  # Specify multiclass classification
        #'tree_method': 'gpu_hist',  # Use GPU accelerated algorithm
        'bosster': 'gdtree',
        'eval_metric': 'rmse',
        'eta': 0.05,
        'max_depth': 13,
        'min_child_weight': 6,
        'gamma': 0,
        'subsample': 0.7,
        'alpha': 0,
        'lamda': 0,
        'updater': "grow_gpu"
        #'max_bin': 16
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(test)


    # Specify sufficient boosting iterations to reach a minimum
    num_round = 4000
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