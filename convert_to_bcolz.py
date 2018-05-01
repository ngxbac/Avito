import json
import pandas as pd
import bcolz
import numpy as np
import os
from tqdm import tqdm
from sys import getsizeof


all_columns = [
    "item_id", "user_id", "region", "city",
    "parent_category_name", "category_name",
    "param_1", "param_2", "param_3", "title",
    "description", "price", "item_seq_number",
    "activation_date", "user_type", "image",
    "image_top_1", "deal_probability"
]

unused_columns = [
    "image", "user_id", "item_id"
]

cat_columns = [
    "region", "city", "parent_category_name",
    "category_name", "param_1", "param_2", "param_3",
    "item_seq_number", "user_type", "image_top_1"
]


config = json.load(open("config.json"))


def load_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["activation_date"])
    df = df.replace(np.nan, -1, regex=True)
    return df


def remove_unused_columns(df: pd.DataFrame, columns=unused_columns):
    for c in columns:
        df = df.drop(c, axis=1)
    return df


def date_to_dow(df: pd.DataFrame, column):
    df[column] = df[column].dt.weekday
    return df


def create_token(df:pd.DataFrame, columns=cat_columns):
    token = []
    for c in columns:
        t = {x: i + 1 for i, x in enumerate(df[c].unique())}
        token.append(t)
    return token


def tokenize_data(df: pd.DataFrame, token, columns=cat_columns):
    token_data = []
    for i, c in enumerate(columns):
        td = np.asarray([token[i].get(key, 0) for key in df[c]], dtype=int)
        token_data.append(td)
    return token_data


def log_prices(df: pd.DataFrame):
    prices = df["price"].as_matrix()
    prices = np.log1p(prices)
    prices[prices==-np.inf] = -1
    return prices


def write_to_bcolz(data, name, root="bcolz_data"):
    if not os.path.exists(root):
        os.mkdir(root)

    fname = f"{root}/{name}"
    # n_dim = len(data)
    # n = len(data[0])
    bcolz_data = bcolz.carray(data, chunklen=1, mode="w", rootdir=fname)
    return bcolz_data


def main():
    # Extract train data
    print("3==D~ Extract train data ")
    print("Read csv file...")
    df = load_csv(config["train_csv"])
    print("Remove unused columns ...")
    df = remove_unused_columns(df)
    print("Convert date to day of week ...")
    df = date_to_dow(df, "activation_date")
    print("Create token ...")
    token = create_token(df)
    print("Tokenize data ...")
    X_train = tokenize_data(df, token)
    print("Make matrix data ...")
    X_train.append(df["activation_date"].as_matrix())
    X_train.append(log_prices(df))
    y_train = df["deal_probability"].as_matrix()
    X_train = np.asarray(X_train).T
    y_train = np.asarray(y_train)
    np.save("X_train.npy", X_train)
    np.save("y_train.npy", y_train)

    # Extract test data
    print("3==D~ Extract test data ")
    print("Read csv file...")
    df = load_csv(config["test_csv"])
    print("Remove unused columns ...")
    df = remove_unused_columns(df)
    print("Convert date to day of week ...")
    df = date_to_dow(df, "activation_date")
    print("Tokenize data ...")
    X_test = tokenize_data(df, token)
    print("Make matrix data ...")
    X_test.append(df["activation_date"].as_matrix())
    X_test.append(log_prices(df))
    X_test = np.asarray(X_test).T
    np.save("X_test.npy", X_test)

    # Save token len
    token_len = [len(t) for t in token]
    print("Save token len ...")
    np.save("token_len.npy", np.asarray(token_len))

if __name__ == '__main__':
    main()