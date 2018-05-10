import warnings
warnings.filterwarnings('ignore')

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
        df[cols] = df[cols].str.replace("[^[:alpha:]]", " ")
        df[cols] = df[cols].str.replace("\\s+", " ")
    return df


def main():
    # Load json config
    config = json.load(open("config.json"))

    txt_vars = [
        "title", "description"
    ]

    extracted_root = config["extracted_features"]

    word_max_dict = config["word_max_dict"]
    word_input_size = config["word_input_size"]
    word_embedding_size = config["word_embedding_size"]

    with utils.timer("Load csv"):
        print("[+] Load csv ...")
        train_df = load_csv(config["train_csv"], txt_vars)
        test_df = load_csv(config["test_csv"], txt_vars)

    n_train = len(train_df)
    df = pd.concat([train_df, test_df])
    del train_df
    del test_df
    gc.collect()

    with utils.timer("Pre-processing"):
        print("\n[+] Preprocessing text")
        df = preprocessing(df, txt_vars)

    with utils.timer("Create token"):
        print("\n[+] Create token")
        tokenizer = text.Tokenizer(num_words=word_max_dict)
        tokenizer.fit_on_texts(df["title"].tolist() + df["description"].tolist())

    for txt in txt_vars:
        with utils.timer(f"Extract {txt}"):
            print(f"\n[+] Extract {txt}")
            list_tokenized = tokenizer.texts_to_sequences(df[txt].tolist())
            X_words = sequence.pad_sequences(list_tokenized, maxlen=word_input_size)
            # Split test and train
            X_train_words = X_words[:n_train, :]
            X_test_words = X_words[n_train:, :]
            # Save the feature
            print(f"[+] Save word features of {txt}")
            utils.save_features(X_train_words, extracted_root, f"X_train_word_{txt}")
            utils.save_features(X_test_words, extracted_root, f"X_test_word_{txt}")

    print("\n[+] Load pretrained embedding")
    # Use pretrained-weights for embedding
    EMBEDDING_FILE = config["fasttext_vec"]
    # embed_size = 300
    embeddings_index = {}
    with open(EMBEDDING_FILE, encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print("[+] Load is done")
    print("[+] Create pretrained embedding")
    word_index = tokenizer.word_index
    # prepare embedding matrix
    num_words = min(word_max_dict, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, word_embedding_size))
    for word, i in word_index.items():
        if i >= word_max_dict:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print("[+] Save pretrained embedding")
    utils.save_features(embedding_matrix, extracted_root, "embedding_weights")

if __name__ == '__main__':
    main()
