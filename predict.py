import json
import pandas as pd
import bcolz
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import models
import datasets as d
import utils
import json


def predict_fold(config, n_folds, X_test, token_len):
    test_dataset = d.NumpyDataset(X_test, None)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    for fold in range(n_folds):
        print("[+] Predict fold: {}".format(fold))
        embedding_size, data_dim = config["embedding_size"], X_test.shape[1]
        n_embedding_layer = len(token_len)
        n_fc_in_features = n_embedding_layer * embedding_size + data_dim - n_embedding_layer

        # Embedding model
        embedding_model = models.AvitorEmbedding(token_len, embedding_size)
        # print(embedding_model)

        # FC model
        model = models.Avitor(embedding_model, n_fc_in_features)
        # print(model)

        if torch.cuda.is_available():
            model.cuda()

        best_val = 0

        model_name = config["model_name"]
        best_ckp = f"ckp_best_{model_name}_{fold}.pth.tar"
        checkpoint_path = f"checkpoint/{model_name}"

        checkpoint_name = f"{checkpoint_path}/{best_ckp}"
        ckp = utils.load_checkpoint(checkpoint_name)
        if ckp:
            model.load_state_dict(ckp["state_dict"])
            best_val = ckp["best_val"]

        print("[+] Model name " + model_name)
        print("[+] Best val {}".format(best_val))

        model.eval()
        preds = []
        for batch_id, (data, _) in enumerate(test_dataloader):
            output = model(data)
            preds += output.data.cpu().numpy().tolist()

        preds = [p[0] for p in preds]
        submission = pd.read_csv(config["sample_submission"])
        submission['deal_probability'] = preds
        submission.to_csv(f"submission_{model_name}_{fold}.csv", index=False)


def predict_one(config, X_test, token_len):
    test_dataset = d.NumpyDataset(X_test, None)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    embedding_size, data_dim = config["embedding_size"], X_test.shape[1]
    n_embedding_layer = len(token_len)
    n_fc_in_features = n_embedding_layer * embedding_size + data_dim - n_embedding_layer

    # Embedding model
    embedding_model = models.AvitorEmbedding(token_len, embedding_size)
    # print(embedding_model)

    # FC model
    model = models.Avitor(embedding_model, n_fc_in_features)
    # print(model)

    if torch.cuda.is_available():
        model.cuda()

    best_val = 0

    model_name = config["model_name"]
    checkpoint_path = f"checkpoint/{model_name}"

    if config["resume"]:
        resume = config["resume"]
        checkpoint_name = f"{checkpoint_path}/{resume}"
        ckp = utils.load_checkpoint(checkpoint_name)
        if ckp:
            model.load_state_dict(ckp["state_dict"])
            start_epoch = ckp["epoch"]
            best_val = ckp["best_val"]

    print("[+] Model name " + model_name)
    print("[+] Best val {}".format(best_val))

    model.eval()

    preds = []
    for batch_id, (data, _) in enumerate(test_dataloader):
        output = model(data)
        preds += output.data.cpu().numpy().tolist()

    preds = [p[0] for p in preds]
    submission = pd.read_csv(config["sample_submission"])
    submission['deal_probability'] = preds
    submission.to_csv(f"submission_{model_name}_one.csv", index=False)

def main():
    # Load json config
    config = json.load(open("config.json"))

    # Load data and token len of embedding layers
    X_test = np.load("X_test.npy")
    token_len = np.load("token_len.npy")

    n_folds = config["n_fold"]
    if n_folds:
        predict_fold(config, n_folds, X_test, token_len)
    else:
        predict_one(config, X_test, token_len)

if __name__ == '__main__':
    main()