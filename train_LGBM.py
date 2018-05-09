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

def load_preds_train_fold(model_name, fold):
    file = f"pred_train_{model_name}_{fold}"
    return utils.load_features("predict_train", file)


def load_preds_test_fold(model_name, fold):
    file = f"predict_root/submission_{model_name}_{fold}.csv"
    preds = pd.read_csv(file)["deal_probability"].tolist()
    return np.array(preds).reshape(-1, 1)

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

    # Load predict train
    n_folds = config["n_fold"]
    model_name = config["model_name"]
    X_pred_train = None
    X_pred_test = None
    for fold in range(n_folds):
        preds_train = load_preds_train_fold(model_name, fold)
        preds_test = load_preds_test_fold(model_name, fold)
        if X_pred_train is None:
            X_pred_train = preds_train
            X_pred_test = preds_test
        else:
            X_pred_train = np.concatenate((X_pred_train, preds_train), axis=1)
            X_pred_test = np.concatenate((X_pred_test, preds_test), axis=1)
    print("Predict train shape {}".format(X_pred_train.shape))
    print("Predict test shape {}".format(X_pred_test.shape))


    # X = np.concatenate((X_train_num, X_train_cat, X_train_desc, X_train_title), axis=1)
    # X = np.concatenate((X_train_num, X_train_cat, X_pred_train), axis=1)
    X = hstack([X_train_num, X_train_cat, X_train_desc, X_train_title])
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)

    # test = np.concatenate((X_test_num, X_test_cat), axis=1)
    # test = np.concatenate((X_test_num, X_test_cat, X_test_desc, X_test_title), axis=1)
    test = hstack([X_test_num, X_test_cat, X_test_desc, X_test_title])

    n_rounds = 200041
    lgbm_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 13,
        # 'num_leaves': 31,
        'feature_fraction': 0.80,
        # 'bagging_fraction': 0.90,
        # 'bagging_freq': 5,
        'learning_rate': 0.05,
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
    submission['deal_probability'].clip(0.0, 1.0, inplace=True)
    submission.to_csv(f"submission_lgbm_boost.csv", index=False)

if __name__ == '__main__':
    main()