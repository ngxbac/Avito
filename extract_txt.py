import os
import pandas as pd
import numpy as np
import glob
import nltk
import gensim
import json
from gensim.models import KeyedVectors


def tokenize(x):
    '''Input: One description'''
    tok=nltk.tokenize.toktok.ToktokTokenizer()
    return [t.lower() for t in tok.tokenize(x)]


def get_vector(model, x):
    '''Input: Single token''' #If the word is out of vocab, then return a 300 dim vector filled with zeros
    try:
        return model.get_vector(x)
    except:
        return np.zeros(shape=300)

def vector_sum(x):
    '''Input:List of word vectors'''
    return np.sum(x,axis=0)


def extract_txt(df, column, prefix="train"):
    features = []
    for desc in df[column].values:
        tokens = tokenize(desc)
        if len(tokens) != 0:  ## If the description is missing then return a 300 dim vector filled with zeros
            word_vecs = [get_vector(w) for w in tokens]
            features.append(vector_sum(word_vecs))
        else:
            features.append(np.zeros(shape=300))

    ## Convert into numpy array
    txt_features = np.array(features)
    np.save(f"X_{prefix}_{coulumn}.npy", txt_features)


def main():
    config = json.load(open("config.json"))
    # Load russian model
    print("[+] Load Russian model ...")
    ru_model = KeyedVectors.load_word2vec_format(config["word2vec_ru"])

    # Load train and test csv
    train_df = pd.read_csv(config["train_csv"])
    test_df = pd.read_csv(config["test_csv"])

    print("[+] Extract train's description text feature ...")
    extract_txt(train_df, "description", "train")
    print("[+] Extract test's description text feature ...")
    extract_txt(test_df, "description", "test")


if __name__ == '__main__':
    main()