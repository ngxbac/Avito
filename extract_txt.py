import pandas as pd
import numpy as np
import json
import gc
import utils

from keras.preprocessing import text, sequence


def load_csv(csv_path, columns):
    df = pd.read_csv(csv_path, usecols=columns)
    return df


def preprocessing(df, columns):
    for cols in columns:
        df[cols] = df[cols].astype(str)
        df[cols] = df[cols].astype(str).fillna(' ')  # FILL NA
        df[cols] = df[cols].str.lower()
    return df


def main():
    # Load json config
    config = json.load(open("config.json"))

    txt_vars = [
        "title", "description"
    ]

    extracted_root = config["extracted_features"]

    max_features = 100000  # 50000
    maxlen = 300

    with utils.timer("Load csv"):
        # print("[+] Load csv ...")
        train_df = load_csv(config["train_csv"], txt_vars)
        test_df = load_csv(config["test_csv"], txt_vars)

    n_train = len(train_df)
    df = pd.concat([train_df, test_df])
    del train_df
    del test_df
    gc.collect()

    with utils.timer("Pre-processing"):
        print("[+] Preprocessing text")
        df = preprocessing(df, txt_vars)

    for txt in txt_vars:
        with utils.timer(f"Extract {txt}"):
            print(f"[+] Extract {txt}")
            tokenizer = text.Tokenizer(num_words=max_features)
            tokenizer.fit_on_texts(df[txt].tolist())
            list_tokenized = tokenizer.texts_to_sequences(df[txt].tolist())
            X_words = sequence.pad_sequences(list_tokenized, maxlen=maxlen)
            X_train_words = X_words[:n_train, :]
            X_test_words = X_words[n_train:, :]
            print(f"[+] Save word features of {txt}")
            utils.save_features(X_train_words, extracted_root, f"X_train_word_{txt}")
            utils.save_features(X_test_words, extracted_root, f"X_test_word_{txt}")


if __name__ == '__main__':
    main()
