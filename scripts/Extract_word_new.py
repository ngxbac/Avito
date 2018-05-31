#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 18:15:32 2018

@author: pooh
"""

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


# root path
root = "/Pooh/Kaggle competition/Kaggle competition/input/"

# For degbug logic
nrows = 100

train_df = pd.read_csv(root+"train_stem.csv",
                       parse_dates=["activation_date"],
                       index_col="item_id", nrows=nrows)
test_df = pd.read_csv(root+"test_stem.csv",
                      parse_dates=["activation_date"],
                      index_col="item_id", nrows=nrows)
txt_vars = [
        "title_stem", "description_stem", "category_name","param_1","param_2"
    ]
word_max_dict = 50000
word_input_size = 100
word_embedding_size = 300

# Merge two dataframes
n_train = len(train_df)
df = pd.concat([train_df, test_df])

##############################################################################
for cols in txt_vars:
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].fillna('')  # FILL NA
    df[cols] = df[cols].str.lower()
        # df[cols] = df[cols].str.replace("[^[:alpha:]]", " ")
    df[cols] = df[cols].str.replace("\\s+", " ")

df['text'] = df.apply(lambda x: " ".join(x[col] for col in txt_vars), axis=1)

###############################################################################

tokenizer = text.Tokenizer(num_words=word_max_dict)
tokenizer.fit_on_texts(df["text"].tolist())


list_tokenized = tokenizer.texts_to_sequences(df["text"].tolist())
X_words = sequence.pad_sequences(list_tokenized, maxlen=word_input_size)
        # Split test and train
X_train_words = X_words[:n_train, :]
X_test_words = X_words[n_train:, :]
        # Save the feature
print("[+] Save word features ")
utils.save_bcolz(X_train_words, "features", f"X_train_word")
utils.save_bcolz(X_test_words, "features", f"X_test_word")

del X_train_words, X_test_words, X_words
gc.collect()

print("\n[+] Load pretrained embedding")
# Use pretrained-weights for embedding
EMBEDDING_FILE = root+"avito.w2v"
model = Word2Vec.load(EMBEDDING_FILE)
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
            embedding_matrix[i] = embedding_vector
    except:
        continue
 # words not found in embedding index will be all-zeros.
print("[+] Save pretrained embedding")
utils.save_bcolz(embedding_matrix, "features", "embedding_weights")

