#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 19:07:14 2018

@author: bac
"""

import warnings
warnings.filterwarnings('ignore')

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
nrows = None
cat_cols = ['user_id', 'region', 'city', 'category_name', "parent_category_name",
                'param_1', 'param_2', 'param_3', 'user_type',
                'weekday', 'ads_count']

##############################################################################
import utils

# Load config
config = json.load(open("config.json"))
root_dir = './'
input_dir = '~/kaggle/avito/'
index_dir = '/home/deeplearning/Kaggle/avito/index'
if not os.path.exists(os.path.join(root_dir + 'lgbm_root')):
    os.mkdir(os.path.join(root_dir + 'lgbm_root'))
lgbm_dir = "lgbm_root"

sub = pd.read_csv(input_dir + "sample_submission.csv", nrows=nrows)
len_sub = len(sub)
print("Sample submission len {}".format(len_sub))

##############################################################################
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
    oof_train = np.zeros((ntr,))
    oof_test = np.zeros((nte,))
    oof_test_skf = np.empty((NFOLDS, nte))

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


def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))


###############################################################################


data_tr1 = pd.read_csv(input_dir + 'train_stem.csv', index_col="item_id",
                       parse_dates=["activation_date"], nrows=nrows)
data_te1 = pd.read_csv(input_dir + 'test_stem.csv', index_col="item_id", parse_dates=["activation_date"], nrows=nrows)
user_df = pd.read_csv(input_dir + 'aggregated_features.csv', nrows=nrows)

data_tr = pd.read_csv(input_dir + 'train.csv', index_col="item_id", parse_dates=["activation_date"], nrows=nrows)

data_te = pd.read_csv(input_dir + 'test.csv', index_col="item_id", parse_dates=["activation_date"], nrows=nrows)

data_tr["description_norm"] = data_tr1["description_stem"]
data_te["description_norm"] = data_te1["description_stem"]

data_tr["title_norm"] = data_tr1["title_stem"]
data_te["title_norm"] = data_te1["title_stem"]

y = data_tr.deal_probability.copy()

ntr = len(data_tr)
nte = len(data_te)
full = pd.concat([data_tr, data_te])
full = full.merge(user_df, on='user_id', how='left')

del data_tr1, data_te1
del user_df
gc.collect()
cols_to_fill = ['description', 'param_1', 'param_2', 'param_3', 'description_norm']
full[cols_to_fill] = full[cols_to_fill].fillna(' ')

# full['params_feat'] = full.apply(lambda row: ' '.join([
#    str(row['param_1']),
#    str(row['param_2']),
#    str(row['param_3'])]), axis=1)

## Tramnsform log
eps = 0.001

full["price"] = np.log(full["price"] + eps)
full["price"].fillna(full["price"].mean(), inplace=True)
full["image_top_1"].fillna(full["image_top_1"].mean(), inplace=True)
full['avg_days_up_user'].fillna(-1, inplace=True)
full['avg_days_up_user'].fillna(-1, inplace=True)
full['avg_times_up_user'].fillna(-1, inplace=True)
full['n_user_items'].fillna(-1, inplace=True)

full['city'] = full['city'] + '_' + full['region']
full['has_image'] = pd.notnull(full.image).astype(int)
full['weekday'] = full['activation_date'].dt.weekday
#full['day_of_month'] = full['activation_date'].dt.day
full['ads_count'] = full.groupby('user_id', as_index=False)['user_id'].transform(lambda s: s.count())
# full.loc[full["item_seq_number"].value_counts()[full["item_seq_number"]].values < 25, "item_seq_number"] = 0
full["item_seq_bin"] = full["item_seq_number"] // 100

full['has_image'] = pd.notnull(full.image).astype(int)
full['has_desc'] = pd.notnull(full.description).astype(int)
full['has_p1'] = pd.notnull(full.param_1).astype(int)
full['has_p2'] = pd.notnull(full.param_2).astype(int)
full['has_p3'] = pd.notnull(full.param_3).astype(int)

# full[col + '_num_chars'] = full[col].apply(len)
textfeats1 = ['description', "title", 'param_1', 'param_2', 'param_3', 'description_norm', "title_norm"]
for col in textfeats1:
    full[col] = full[col].astype(str)
    full[col] = full[col].astype(str).fillna(' ')
    full[col] = full[col].str.lower()

textfeats2 = ['description', "title"]
for col in textfeats2:
    full[col + '_num_words'] = full[col].apply(lambda s: len(s.split()))
    full[col + '_num_unique_words'] = full[col].apply(lambda s: len(set(w for w in s.split())))
    full[col + '_words_vs_unique'] = full[col + '_num_unique_words'] / full[col + '_num_words'] * 100
    full[col + '_num_lowE'] = full[col].str.count("[a-z]")
    full[col + '_num_lowR'] = full[col].str.count("[а-я]")
    full[col + '_num_pun'] = full[col].str.count("[[:punct:]]")
    full[col + '_num_dig'] = full[col].str.count("[[:digit:]]")
print('Creating difference in num_words of description and title')
#full['diff_num_word_des_title'] = full['description_num_words'] - full['title_num_words']
full['param_2'] = full['param_1'] + ' ' + full['param_2']
# full['param_1'] = full['param_3']
full['param_3'] = full['param_2'] + ' ' + full['param_3']


###############################################################################
full['params'] = full['param_3']+' '+full['title_norm']
full['text']=full['description_norm']+ ' ' + full['title_norm']
###############################################################################

names = ["city", "param_1", "user_id"]
for i in names:
    full.loc[full[i].value_counts()[full[i]].values < 100, i] = "Rare_value"

full.loc[full["image_top_1"].value_counts()[full["image_top_1"]].values < 150, "image_top_1"] = -1
full.loc[full["item_seq_number"].value_counts()[full["item_seq_number"]].values < 150, "item_seq_number"] = -1
###############################################################################

for c in cat_cols:
    le = LabelEncoder()
    allvalues = np.unique(full[c].values).tolist()
    le.fit(allvalues)
    full[c] = le.transform(full[c].values)

###############################################################################

data_tr = full[:ntr]
data_te = full[ntr:]

del full
gc.collect()


###############################################################################


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
            gp = df.groupby(c)[['deal_probability', 'price']]
            desc = gp.describe()
            self._stats[c] = desc[[('deal_probability', 'mean'), ('deal_probability', 'std'),
                                   ('price', 'mean'), ('price', 'std')]]

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

            df[c + '_to_price'] = df.price / df[c + '_price_mean']
            df[c + '_to_price'] = df[c + '_to_price'].fillna(1.0)

    def fit_transform(self, df):
        '''
        First learn the feature statistics, then add them to the dataframe.
        '''
        self.fit(df)
        self.transform(df)


fStats = FeaturesStatistics(['region', 'city', 'parent_category_name', 'category_name',
                             'param_1', 'param_2', 'param_3', 'user_type', 'image_top_1',
                             'ads_count', 'weekday'])

###############################################################################

russian_stop = set(stopwords.words('russian'))

titles_tfidf = TfidfVectorizer(
    stop_words=russian_stop,
    max_features=20000,
    norm='l2',
    sublinear_tf=True,
    smooth_idf=False,
    dtype=np.float32,
)

tr_titles = titles_tfidf.fit_transform(data_tr.params).astype('float32')
te_titles = titles_tfidf.transform(data_te.params).astype('float32')

desc_tfidf = TfidfVectorizer(
    stop_words=russian_stop,
    max_features=20000,
    norm='l2',
    sublinear_tf=True,
    smooth_idf=False,
    dtype=np.float32,
)

tr_desc = desc_tfidf.fit_transform(data_tr.text)
te_desc = desc_tfidf.transform(data_te.text)

###############################################################################
from sklearn.metrics import mean_squared_error
from math import sqrt

kf = KFold(ntr, n_folds=NFOLDS, shuffle=True, random_state=SEED)

ridge_params = {'alpha': 2.0, 'fit_intercept': True, 'normalize': False, 'copy_X': True,
                'max_iter': None, 'tol': 0.001, 'solver': 'auto', 'random_state': SEED}

# Ridge oof method from Faron's kernel

ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
ridge_oof_train_desc, ridge_oof_test_desc = get_oof(ridge, tr_desc, y, te_desc)
ridge_oof_train_title, ridge_oof_test_title = get_oof(ridge, tr_titles, y, te_titles)
#ridge_oof_train_params, ridge_oof_test_params = get_oof(ridge, tr_params, y, te_params)

rms = sqrt(mean_squared_error(y, ridge_oof_train_desc))
print('Ridge OOF RMSE: {}'.format(rms))

print("Modeling Stage")

data_tr['ridge_oof_desc'] = ridge_oof_train_desc
data_te['ridge_oof_desc'] = ridge_oof_test_desc

data_tr['ridge_preds_title'] = ridge_oof_train_title
data_te['ridge_preds_title'] = ridge_oof_test_title


del  ridge_oof_train_title, ridge_oof_test_title, ridge_oof_train_desc, ridge_oof_test_desc
gc.collect()

###############################################################################

n_train_folds = 5
kf_train = KFold(ntr, n_folds=n_train_folds, shuffle=True, random_state=2018)
for fold, (train_index, val_index) in enumerate(kf_train):

    print(f"\n[+] Fold {fold}")

    data_te_fold = data_te

    data_tr_fold = data_tr.iloc[train_index]
    data_va_fold = data_tr.iloc[val_index]

    fStats.fit_transform(data_tr_fold)
    fStats.transform(data_va_fold)
    fStats.transform(data_te_fold)

    tr_titles = titles_tfidf.fit_transform(data_tr_fold.params).astype('float32')
    va_titles = titles_tfidf.transform(data_va_fold.params).astype('float32')
    te_titles = titles_tfidf.transform(data_te_fold.params).astype('float32')

    tr_desc = desc_tfidf.fit_transform(data_tr_fold.text).astype('float32')
    va_desc = desc_tfidf.transform(data_va_fold.text).astype('float32')
    te_desc = desc_tfidf.transform(data_te_fold.text).astype('float32')

    columns_to_drop = ['title', 'description', 'params', 'image',
                       'activation_date', 'deal_probability', 'title_norm', 'description_norm', 'text']

    X_tr = hstack([csr_matrix(data_tr_fold.drop(columns_to_drop, axis=1)), tr_titles, tr_desc])
    y_tr = data_tr_fold['deal_probability']
    del tr_titles, tr_desc, data_tr_fold

    gc.collect()
    X_va = hstack([csr_matrix(data_va_fold.drop(columns_to_drop, axis=1)), va_titles, va_desc])
    y_va = data_va_fold['deal_probability']
    del va_titles, va_desc, data_va_fold
    gc.collect()
    X_te = hstack([csr_matrix(data_te_fold.drop(columns_to_drop, axis=1)), te_titles, te_desc])

    del te_titles, te_desc, data_te_fold

    gc.collect()

###############################################################################

    X_tr_lgb = lgb.Dataset(X_tr, label=y_tr)
    X_va_lgb = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, reference=X_tr_lgb)

    parameters = {
        'objective': 'regression',
        'boosting': 'dart',
        'metric': 'rmse',
        'num_leaves': 255,
        'learning_rate': 0.02,
        'feature_fraction': 0.4,
        'bagging_fraction': 0.8,
        'bagging_freq': 2,
        'num_threads': 12,
    }

    model = lgb.train(parameters, X_tr_lgb, valid_sets=[X_tr_lgb, X_va_lgb],
                      valid_names=['train','valid'],
                      num_boost_round=7000, early_stopping_rounds=300, verbose_eval=20)

    y_pred = model.predict(X_te)
    y_pred_round = y_pred
    y_pred_round[y_pred_round < 0.005] = 0
    y_pred_round[y_pred_round > 0.95] = 1
    sub['deal_probability'] = y_pred
    sub_round = sub
    sub_round['deal_probability'] = y_pred_round
    # sub['deal_probability'].clip(0.0, 1.0, inplace=True)
    if not os.path.exists(os.path.join(root_dir + 'submission')):
        os.mkdir(os.path.join(root_dir + 'submission'))
    sub.to_csv(root_dir + f'/submission/lgb_team_dart_{fold}.csv', index=False)
    sub_round.to_csv(root_dir + f'/submission/lgb_team_round_dart_{fold}.csv', index=False)
    print('Finished saving submision csv file')
    