#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 19:07:14 2018

@author: pooh
"""

import gc
import lightgbm as lgb
# import xgboost as xgb
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

# from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold
import utils
import argparse
import json


# import string

##############################################################################
not_important = ['param_2_num_words', 'param_3_num_chars', 'weekend', 'param_1_num_words', 'weekday', 'title_4', 'title_1']
NFOLDS = 5
SEED = 42
nrows = 100
USE_TFIDF = False
cat_cols = ["image_top_1",'region', 'city', 'category_name', "parent_category_name",
                'param_1', 'param_2', 'param_3', 'user_type',
                'weekday']

columns_to_drop = ['title', 'description', 'image', "p1_price_mean", "user_id",
                   'activation_date']

class FeaturesStatistics():
    def __init__(self, cols):
        self._stats = None
        self._agg_cols = cols

    def fit(self, df):
        '''
        Compute the mean and std of some features from a given data frame
        '''
        self._stats = {}

        # For each feature to be aggregated
        for c in tqdm(self._agg_cols, total=len(self._agg_cols)):
            # Compute the mean and std of the deal prob and the price.
            gp = df.groupby(c)[['deal_probability', 'price', 'item_seq_number', "image_top_1"]]
            desc = gp.describe()
            self._stats[c] = desc[[('deal_probability', 'mean'), ('deal_probability', 'std'),
                                   ('price', 'mean'), ('price', 'std'),
                                   ('item_seq_number', 'mean'), ('item_seq_number', 'std'),
                                   ("image_top_1", "count")]]

    def transform(self, df):
        '''
        Add the mean features statistics computed from another dataset.
        '''
        # For each feature to be aggregated
        for c in tqdm(self._agg_cols, total=len(self._agg_cols)):
            # Add the deal proba and price statistics corrresponding to the feature
            df[c + '_dp_mean'] = df[c].map(self._stats[c][('deal_probability', 'mean')])
            df[c + '_dp_std'] = df[c].map(self._stats[c][('deal_probability', 'std')])
            df[c + '_price_mean'] = df[c].map(self._stats[c][('price', 'mean')])
            df[c + '_price_std'] = df[c].map(self._stats[c][('price', 'std')])

            df[c + '_seq_mean'] = df[c].map(self._stats[c][('item_seq_number', 'mean')])
            df[c + '_seq_std'] = df[c].map(self._stats[c][('item_seq_number', 'std')])

            df[c + '_to_price'] = df.price / df[c + '_price_mean']
            df[c + '_to_price'] = df[c + '_to_price'].fillna(1.0)

            if c != "image_top_1":
                df[c + "_count_img_top1"] = df[c].map(self._stats[c][('image_top_1', 'count')])

    def fit_transform(self, df):
        '''
        First learn the feature statistics, then add them to the dataframe.
        '''
        self.fit(df)
        self.transform(df)


fStats = FeaturesStatistics(['region', 'city', 'parent_category_name', 'category_name',
                             'param_1', 'param_2', 'param_3', 'user_type',
                             'weekday'])

import string
##############################################################################
import utils

parser = argparse.ArgumentParser()
parser.add_argument('nrows', type=int)
args = parser.parse_args()

nrows = args.nrows
if nrows < 0:
    nrows = None

# Load config
config = json.load(open("config.json"))
root_dir = './'
input_dir = '/home/deeplearning/Kaggle/avito/input/'
index_dir = '/home/deeplearning/Kaggle/avito/index'
if not os.path.exists(os.path.join(root_dir + 'lgbm_root')):
    os.mkdir(os.path.join(root_dir + 'lgbm_root'))
lgbm_dir = "lgbm_root"

sub = pd.read_csv(config["sample_submission"], nrows=nrows)
len_sub = len(sub)
print("Sample submission len {}".format(len_sub))


def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

user_df = pd.read_csv(input_dir + 'aggregated_features.csv', nrows=nrows)
train = pd.read_csv(input_dir + 'train.csv', index_col="item_id", parse_dates=["activation_date"], nrows=nrows)
test = pd.read_csv(input_dir + 'test.csv', index_col="item_id", parse_dates=["activation_date"], nrows=nrows)
train = train.merge(user_df, on='user_id', how='left')
test = test.merge(user_df, on='user_id', how='left')
agg_cols = list(user_df.columns)[1:]


y = train.deal_probability.copy()


for df in [train, test]:
    #df['has_image'] = pd.notnull(df.image).astype(int)
    #df['has_desc'] = pd.notnull(df.description).astype(int)
    #df['has_p1'] = pd.notnull(df.param_1).astype(int)
    #df['has_p2'] = pd.notnull(df.param_2).astype(int)
    #df['has_p3'] = pd.notnull(df.param_3).astype(int)

    df['description'].fillna('', inplace=True)
    df['title'].fillna('', inplace=True)

    df['weekday'] = pd.to_datetime(df['activation_date']).dt.weekday

    for col in ['description', 'title']:
        df['num_words_' + col] = df[col].apply(lambda comment: len(comment.split()))
        df['num_unique_words_' + col] = df[col].apply(lambda comment: len(set(w for w in comment.split())))
        df['words_vs_unique_' + col] = df['num_unique_words_' + col] / df['num_words_' + col] * 100
        df[col + '_num_lowE'] = df[col].str.count("[a-z]")
        df[col + '_num_lowR'] = df[col].str.count("[а-я]")
        df[col + '_num_pun'] = df[col].str.count("[[:punct:]]")
        df[col + '_num_dig'] = df[col].str.count("[[:digit:]]")

    # df['city'] = df['region'] + '_' + df['city']


for col in agg_cols:
    train[col].fillna(-1, inplace=True)
    test[col].fillna(-1, inplace=True)

p1_mean_price = train.groupby("param_1")["price"].agg("mean").reset_index()
p1_mean_price.columns = ["param_1", "p1_price_mean"]
train = pd.merge(train, p1_mean_price, how='left', on='param_1')

train["price"].fillna((train["p1_price_mean"]), inplace=True)
train["price"] = np.log(train["price"] + 0.001)
train["price"].fillna(-999, inplace=True)


p1_mean_price = test.groupby("param_1")["price"].agg("mean").reset_index()
p1_mean_price.columns = ["param_1", "p1_price_mean"]
test = pd.merge(test, p1_mean_price, how='left', on='param_1')

test["price"].fillna((test["p1_price_mean"]), inplace=True)
test["price"] = np.log(test["price"] + 0.001)
test["price"].fillna(-999, inplace=True)


for feature in cat_cols:
    print(f'Transforming {feature}...')
    encoder = LabelEncoder()
    train[feature].fillna('unknown', inplace=True)
    test[feature].fillna('unknown', inplace=True)
    encoder.fit(train[feature].append(test[feature]).astype(str))

    train[feature] = encoder.transform(train[feature].astype(str))
    test[feature] = encoder.transform(test[feature].astype(str))

fStats.fit_transform(train)
fStats.transform(test)

if USE_TFIDF:
    ###############################################################################
    count_vectorizer_title = CountVectorizer(stop_words=stopwords.words('russian'), lowercase=True, min_df=2)

    title_counts = count_vectorizer_title.fit_transform(train['title'].append(test['title']))

    train_title_counts = title_counts[:len(train)]
    test_title_counts = title_counts[len(train):]


    count_vectorizer_desc = TfidfVectorizer(stop_words=stopwords.words('russian'),
                                            lowercase=True, ngram_range=(1, 2),
                                            max_features=17000)

    desc_counts = count_vectorizer_desc.fit_transform(train['description'].append(test['description']))

    train_desc_counts = desc_counts[:len(train)]
    test_desc_counts = desc_counts[len(train):]

    train_title_counts.shape, train_desc_counts.shape

    ##############################################################################
    NFOLDS = 5
    SEED = 42
    ntrain = len(train)
    ntest = len(test)

    kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)
    class SklearnWrapper(object):
        def __init__(self, clf, seed=0, params=None, seed_bool=True):
            if (seed_bool == True):
                params['random_state'] = seed
            self.clf = clf(**params)

        def train(self, x_train, y_train):
            self.clf.fit(x_train, y_train)

        def predict(self, x):
            return self.clf.predict(x)


    def get_oof(clf, x_train, y, x_test):
        oof_train = np.zeros((ntrain,))
        oof_test = np.zeros((ntest,))
        oof_test_skf = np.empty((NFOLDS, ntest))

        for i, (train_index, test_index) in enumerate(kf):
            print('\nFold {}'.format(i))
            x_tr = x_train[train_index]
            y_tr = y[train_index]
            x_te = x_train[test_index]

            clf.train(x_tr, y_tr)

            oof_train[test_index] = clf.predict(x_te)
            oof_test_skf[i, :] = clf.predict(x_test)

        oof_test[:] = oof_test_skf.mean(axis=0)
        return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

    from math import sqrt

    ridge_params = {'alpha':20.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                    'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}

    #Ridge oof method from Faron's kernel
    #I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
    #It doesn't really add much to the score, but it does help lightgbm converge faster
    ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
    ridge_oof_train_desc, ridge_oof_test_desc = get_oof(ridge, train_desc_counts, y, test_desc_counts)
    ridge_oof_train_title, ridge_oof_test_title = get_oof(ridge, train_title_counts, y, test_title_counts)

    train["ridge_desc"] = ridge_oof_train_desc
    test["ridge_desc"] = ridge_oof_test_desc

    train["ridge_title"] = ridge_oof_train_title
    test["ridge_title"] = ridge_oof_test_title

    del ridge_oof_train_desc, ridge_oof_test_desc
    del ridge_oof_train_title, ridge_oof_test_title

###############################################################################
indicates = range(len(train))
data_tr, data_va, train_index, val_index = train_test_split(train, indicates, shuffle=True, test_size=0.1, random_state=42)

all_columns = data_tr.columns

train_columns = list(set(all_columns) -  set(columns_to_drop) - set(["deal_probability"]))
# train_columns = list(set(all_columns) -  set(columns_to_drop))

if USE_TFIDF:
    X_tr = hstack([csr_matrix(data_tr.drop(columns_to_drop + ["deal_probability"], axis=1)), train_desc_counts[train_index], train_title_counts[train_index]])
else:
    X_tr = data_tr.drop(columns_to_drop + ["deal_probability"], axis=1)
y_tr = data_tr['deal_probability']
del data_tr, train

gc.collect()
if USE_TFIDF:
    X_va = hstack([csr_matrix(data_va.drop(columns_to_drop + ["deal_probability"], axis=1)), train_desc_counts[val_index], train_title_counts[val_index]])
else:
    X_va = data_va.drop(columns_to_drop + ["deal_probability"], axis=1)
y_va = data_va['deal_probability']
del data_va
gc.collect()

if USE_TFIDF:
    X_te = hstack([csr_matrix(test.drop(columns_to_drop, axis=1)), test_desc_counts, test_title_counts])
else:
    X_te = test.drop(columns_to_drop, axis=1)

if USE_TFIDF:
    train_columns = np.hstack([train_columns, count_vectorizer_desc.get_feature_names(), count_vectorizer_title.get_feature_names()])

del test

gc.collect()

X_tr_lgb = lgb.Dataset(X_tr, label=y_tr, feature_name=list(train_columns))
X_va_lgb = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=X_tr_lgb, feature_name=list(train_columns))

parameters = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 255,
    'learning_rate': 0.02,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.8,
    #'bagging_freq': 2
}

model = lgb.train(parameters, X_tr_lgb, valid_sets=[X_tr_lgb, X_va_lgb],
                  valid_names=['train','valid'],
                  num_boost_round=7000, early_stopping_rounds=200, verbose_eval=100)


y_pred = model.predict(X_te)
y_pred[y_pred < 0.005] = 0
y_pred[y_pred > 0.95] = 1
sub['deal_probability'] = y_pred
# sub['deal_probability'].clip(0.0, 1.0, inplace=True)
if not os.path.exists(os.path.join(root_dir + 'submission')):
    os.mkdir(os.path.join(root_dir + 'submission'))
sub.to_csv(root_dir + '/submission/lgb_team.csv', index=False)
print('Finished saving submision csv file')

y_pred = model.predict(X_tr)
y_pred[y_pred < 0.005] = 0
y_pred[y_pred > 0.95] = 1
if not os.path.exists(os.path.join(root_dir + 'preds')):
    os.mkdir(os.path.join(root_dir + 'preds'))
np.save(root_dir + '/preds/pred_train.npy',y_pred)
print('Finished saving pred train')


y_pred = model.predict(X_va)
y_pred[y_pred < 0.005] = 0
y_pred[y_pred > 0.95] = 1
np.save(root_dir + '/preds/pred_validation.npy',y_pred)
print('Finished saving pred validation')

# # Viz
import seaborn as sns
import matplotlib.pyplot as plt
# Feature Importance Plot
f, ax = plt.subplots(figsize=[20,20])
lgb.plot_importance(model, max_num_features=200, ax=ax)
plt.title("Light GBM Feature Importance")
plt.savefig('lgb_team_feature_import.png')

##############################################################
if not os.path.exists(os.path.join(root_dir + 'saved_model')):
    os.mkdir(os.path.join(root_dir + 'saved_model'))
tr_best_score = model.best_score['train']['rmse']
val_best_score = model.best_score['valid']['rmse']
model_path = os.path.join(root_dir,'saved_model/lbgm_' + str(round(tr_best_score, 4)) +'_' + str(round(val_best_score, 4)) + '.txt')
model.save_model(model_path)
##############################################################