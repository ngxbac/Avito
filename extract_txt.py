import os
import pandas as pd
import numpy as np
import glob
import nltk
import gensim
import json
import gc
import utils

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer # For text feature
from nltk.corpus import stopwords # identify stopwords
stopWords = stopwords.words('russian')


def load_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["activation_date"])
    # df = df.replace(np.nan, -1, regex=True)
    return df


def extract_TFIDF(df, columns):
    tfidfs = []
    for c in columns:
        tfidf_vector = TfidfVectorizer(max_features=5000, stop_words = stopWords)
        # Fill-in missing values
        df[c] = df[c].fillna(' ')
        # fit and transform Russian
        tfidf_vector.fit(df[c])
        tfidf = tfidf_vector.transform(df['description'])
        tfidfs.append(tfidf)
    gc.collect()
    return tfidfs


def extract_SVD(tfidfs, n_comp=3):
    svds = []
    for tfidf in tfidfs:
        svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
        svd_obj.fit(tfidf)
        svd = svd_obj.transform(tfidf)
        svds.append(svd)
    return svds


def main():
    # Load json config
    config = json.load(open("config.json"))

    with utils.timer("Load csv"):
        # print("[+] Load csv ...")
        train_df = load_csv(config["train_csv"])
        test_df = load_csv(config["test_csv"])

    df = pd.concat([train_df, test_df])
    del train_df
    del test_df
    gc.collect()

    txt_vars = [
        "title", "description"
    ]

    for txt in txt_vars:
        df[txt] = df[txt].astype("str")

    with utils.timer("Extract TFIDF ..."):
        tfidfs = extract_TFIDF(df, txt_vars)

    with utils.timer("Extract SVD ..."):
        svds = extract_SVD(tfidfs)

    extracted_root = config["extracted_features"]
    utils.save_features(np.asarray(tfidfs, dtype=np.float32), extracted_root, "TFIDF")
    utils.save_features(np.asarray(svds, dtype=np.float32), extracted_root, "SVD")


if __name__ == '__main__':
    main()