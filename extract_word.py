import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json
import gc
import utils
from keras.preprocessing import text, sequence
import re
from gensim.models import KeyedVectors, Word2Vec
import argparse


nrows = 1000
# Parse the argument
def add_args(parser):
    arg = parser.add_argument
    arg('--embed', type=str, default='selftrain_300', help='use selftrain or fasttext')
    arg('--text', type=str, default='default', help='use default, norm or stem text')

parser = argparse.ArgumentParser()
arg = parser.add_argument

add_args(parser)
args = parser.parse_args()

# Load json config
config = json.load(open("config.json"))
root = config["root"]

if args.text == "default":
    train_csv = "train.csv"
    test_csv = "test.csv"
    desc_name = "description"
    title_name = "title"
elif args.text == "norm":
    train_csv = "train_norm.csv"
    test_csv = "test_norm.csv"
    desc_name = "description_norm"
    title_name = "title_norm"
elif args.text == "stem":
    train_csv = "train_stem.csv"
    test_csv = "test_stem.csv"
    desc_name = "description_stem"
    title_name = "title_stem"
else:
    exit()

train_csv = f"{root}/{train_csv}"
test_csv = f"{root}/{test_csv}"

if args.embed == "selftrain_300":
    pretrained_path = "/Self-training-embeding_300/avito.w2v"
elif args.embed == "selftrain_100":
    pretrained_path = "/Self-training-embeding_100/avito.w2v"
elif args.embed == "fasttext":
    pretrained_path = "wiki.ru.vec"
else:
    exit()

embedding_path = f"{root}/{pretrained_path}"

def load_csv(csv_path, columns):
    df = pd.read_csv(csv_path, usecols=columns, nrows=nrows)
    return df


def preprocessing(df, columns):
    for cols in columns:
        df[cols] = df[cols].astype(str)
        df[cols] = df[cols].fillna('')  # FILL NA
        df[cols] = df[cols].str.lower()
        # df[cols] = df[cols].str.replace("[^[:alpha:]]", " ")
        df[cols] = df[cols].str.replace("\\s+", " ")

    return df


def main():

    txt_vars = [
        title_name, desc_name
    ]

    extracted_root = config["features"]
    extracted_root = f"{extracted_root}/{args.text}_{args.embed}"

    word_max_dict = config["word_max_dict"]
    word_input_size = config["word_input_size"]
    word_embedding_size = config["word_embedding_size"]

    with utils.timer("Load csv"):
        print("[+] Load csv ...")
        train_df = load_csv(train_csv, txt_vars)
        test_df = load_csv(test_csv, txt_vars)

    n_train = len(train_df)
    df = pd.concat([train_df, test_df])
    del train_df
    del test_df
    gc.collect()

    with utils.timer("Pre-processing"):
        print("\n[+] Preprocessing text")
        df = preprocessing(df, txt_vars)

    df['text'] = df.apply(lambda x: " ".join(x[col] for col in txt_vars), axis=1)

    with utils.timer("Create token"):
        print("\n[+] Create token")
        tokenizer = text.Tokenizer(num_words=word_max_dict)
        tokenizer.fit_on_texts(df["text"].tolist())

    with utils.timer("Extract word"):
        print("\n[+] Extract word")
        list_tokenized = tokenizer.texts_to_sequences(df["text"].tolist())
        X_words = sequence.pad_sequences(list_tokenized, maxlen=word_input_size)
        # Split test and train
        X_train_words = X_words[:n_train, :]
        X_test_words = X_words[n_train:, :]
        # Save the feature
        print("[+] Save word features ")
        utils.save_bcolz(X_train_words, extracted_root, f"X_train_word")
        utils.save_bcolz(X_test_words, extracted_root, f"X_test_word")

    del X_train_words, X_test_words, X_words
    gc.collect()

    print("\n[+] Load pretrained embedding")
    # Use pretrained-weights for embedding
    if args.embed == "fasttext":
        model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
    else:
        model = Word2Vec.load(embedding_path)
    print("[+] Load is done")
    print("[+] Create pretrained embedding")
    word_index = tokenizer.word_index
    # prepare embedding matrix
    num_words = min(word_max_dict, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, word_embedding_size))
    for word, i in word_index.items():
        try:
            embedding_vector = model[word]
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        except:
            continue

    print("[+] Save pretrained embedding")
    utils.save_bcolz(embedding_matrix, extracted_root, "embedding_weights")


if __name__ == '__main__':
    main()
