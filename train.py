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
from sklearn.model_selection import StratifiedKFold

import models
import datasets as d
import utils
import json


class rmse(nn.Module):
    def __init__(self):
        super(rmse, self).__init__()

    def forward(self, y, y_hat):
        return torch.sqrt(torch.mean((y-y_hat).pow(2)))


def train_normal(config, X, y, token_len):
    # Create train/valid dataset and dataloader
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=1700)
    train_dataset = d.NumpyDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    valid_dataset = d.NumpyDataset(X_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=True)

    embedding_size, data_dim = config["embedding_size"], X_train.shape[1]
    n_embedding_layer = len(token_len)
    n_fc_in_features = n_embedding_layer * embedding_size + data_dim - n_embedding_layer

    # Embedding model
    embedding_model = models.AvitorEmbedding(token_len, embedding_size)
    # print(embedding_model)

    # FC model
    model = models.Avitor(embedding_model, n_fc_in_features)
    # print(model)

    # MSE loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    start_epoch = 0
    best_val = float("inf")
    is_best = False
    patience = 0

    model_name = config["model_name"]
    epoch_ckp = f"ckp_{model_name}.pth.tar"
    best_ckp = f"ckp_best_{model_name}.pth.tar"
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
    print("[+] Start at epoch {}".format(start_epoch))
    print("[+] Best val: {}".format(best_val))
    print("[+] Number of embedding layer: {}".format(n_embedding_layer))
    print("[+] Total features: {}".format(data_dim))

    for epoch in range(config["epoch"]):
        utils.train(epoch, train_dataloader, model, criterion, optimizer)
        val_loss = utils.test(epoch, valid_dataloader, model, criterion)
        if val_loss < best_val:
            best_val = val_loss
            is_best = True
            patience = 0
        else:
            is_best = False
            patience += 1

        if patience >= config["patience"]:
            print("[+] Early stopping !!!")
            print("[+] Best val {}".format(best_val))
            return

        utils.save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_val": best_val
        }, is_best, model_name, epoch_ckp, best_ckp)


def train_fold(config, n_folds, X, y, token_len):
    skf = StratifiedKFold(n_folds)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print("[+] Fold: {}".format(fold))
        X_train = X[train_index]
        y_train = y[train_index]
        train_dataset = d.NumpyDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

        X_valid = X[test_index]
        y_valid = y[test_index]
        valid_dataset = d.NumpyDataset(X_valid, y_valid)
        valid_dataloader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=True)

        embedding_size, data_dim = config["embedding_size"], X_train.shape[1]
        n_embedding_layer = len(token_len)
        n_fc_in_features = n_embedding_layer * embedding_size + data_dim - n_embedding_layer
        # Embedding model
        embedding_model = models.AvitorEmbedding(token_len, embedding_size)
        # print(embedding_model)

        # FC model
        model = models.Avitor(embedding_model, n_fc_in_features)
        # print(model)

        # MSE loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
        if torch.cuda.is_available():
            model.cuda()
            criterion.cuda()

        start_epoch = 0
        best_val = float("inf")
        is_best = False
        patience = 0

        model_name = config["model_name"]
        epoch_ckp = f"ckp_{model_name}_{fold}.pth.tar"
        best_ckp = f"ckp_best_{model_name}_{fold}.pth.tar"
        checkpoint_path = f"checkpoint/{model_name}"

        for epoch in range(config["epoch"]):
            utils.train(epoch, train_dataloader, model, criterion, optimizer)
            val_loss = utils.test(epoch, valid_dataloader, model, criterion)
            if val_loss < best_val:
                best_val = val_loss
                is_best = True
                patience = 0
            else:
                is_best = False
                patience += 1

            if patience >= config["patience"]:
                print("[+] Early stopping !!!")
                print("[+] Best val {}".format(best_val))
                return

            utils.save_checkpoint({
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_val": best_val
            }, is_best, model_name, epoch_ckp, best_ckp)


def main():
    # Load json config
    config = json.load(open("config.json"))

    # Load data and token len of embedding layers
    X = np.load("X_train.npy")
    y = np.load("y_train.npy")
    token_len = np.load("token_len.npy")

    n_folds = config["n_fold"]
    if n_folds:
        train_fold(config, n_folds, X, y, token_len)
    else:
        train_normal(config, X, y, token_len)

if __name__ == '__main__':
    main()