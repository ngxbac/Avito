import json
import pandas as pd
import bcolz
import numpy as np
import os

import lightgbm as lgb

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


    #X = np.concatenate((X_train_num, X_train_cat, X_train_desc, X_train_title), axis=1)
    #X = np.concatenate((X_train_num, X_train_cat), axis=1)
    X = hstack([X_train_num, X_train_cat, X_train_param])
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

    # test = np.concatenate((X_test_num, X_test_cat), axis=1)
    # test = np.concatenate((X_test_num, X_test_cat, X_test_desc, X_test_title), axis=1)
    test = hstack([X_test_num, X_test_cat, X_test_param])

    n_rounds = 200041
    lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 13,
        'num_leaves': 31,
        'feature_fraction': 0.80,
        # 'bagging_fraction': 0.90,
        # 'bagging_freq': 5,
        'learning_rate': 0.03,
        'lambda_l2': 5,
        'verbose': 0,
        "device": "gpu",
        "max_bin": 128,
        "num_threads": 4
    }

    lgtrain = lgb.Dataset(X_train, label=y_train)
    lgvalid = lgb.Dataset(X_valid, label=y_valid)
    # lgbtest = lgb.Dataset(test)


    # Specify sufficient boosting iterations to reach a minimum
    num_round = 4000
    gpu_res = {}  # Store accuracy result
    # Train model
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=n_rounds,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train', 'valid'],
        early_stopping_rounds=200,
        verbose_eval=50
    )

    deal_probability = lgb_clf.predict(test)
    submission = pd.read_csv(config["sample_submission"])
    submission['deal_probability'] = deal_probability
    submission.to_csv(f"submission_lgbm.csv", index=False)

if __name__ == '__main__':
    main()