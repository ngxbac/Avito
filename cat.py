# Catboost for Avito Demand Prediction Challenge
# https://www.kaggle.com/c/avito-demand-prediction
# By Nick Brooks, April 2018

import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

# Viz
import seaborn as sns
import matplotlib.pyplot as plt

import json
config = json.load(open("config.json"))

print("\nData Load Stage")
training = pd.read_csv(config["train_csv"], index_col = "item_id", parse_dates = ["activation_date"])
training = training.replace(np.nan, -1, regex=True)
traindex = training.index
testing = pd.read_csv(config["test_csv"], index_col = "item_id", parse_dates = ["activation_date"])
testing = testing.replace(np.nan, -1, regex=True)
testdex = testing.index
y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

# Combine Train and Test
df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()
print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))

print("Feature Engineering")
df["price"] = np.log(df["price"]+0.001)
df["price"].fillna(-999,inplace=True)
df["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday

print("\nEncode Variables")
categorical = ["region","city","parent_category_name","category_name","user_type","image_top_1", "param_1", "param_2", "param_3"]
text = ["description", "title"]
print("Encoding :",categorical)

def extract_text_features_as_numeric(df, columns):
    for cols in columns:
        df[cols] = df[cols].astype(str)
        df[cols] = df[cols].fillna('NA')  # FILL NA
        df[cols] = df[cols].str.lower()  # Lowercase all text, so that capitalized words dont get treated differently
        df[cols + '_num_chars'] = df[cols].apply(len)  # Count number of Characters
        df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split()))  # Count number of Words
        df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        df[cols + '_words_vs_unique'] = df[cols + '_num_unique_words'] / df[cols + '_num_words'] * 100  # Count Unique Words

    return df

df = extract_text_features_as_numeric(df, text)

print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")
russian_stop = set(stopwords.words('russian'))

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}


def title_features(df, n_comp=3):
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, **tfidf_para)
    tfidf_vec.fit(df['title'].values.tolist())
    tfidf = tfidf_vec.transform(df['title'].values.tolist())


    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(tfidf)
    svd = svd_obj.transform(tfidf)
    svd_df = pd.DataFrame(svd)
    svd_df.columns = ['svd_title_' + str(i + 1) for i in range(n_comp)]
    print("Title SVD_DF")
    print(svd_df.head(5))

    df = pd.concat([df, svd_df], axis=1)
    return df


def description_features(df, n_comp=3):
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, **tfidf_para)
    tfidf_vec.fit(df['description'].values.tolist())
    tfidf = tfidf_vec.transform(df['description'].values.tolist())

    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(tfidf)
    svd = svd_obj.transform(tfidf)
    svd_df = pd.DataFrame(svd)
    svd_df.columns = ['svd_desc_' + str(i + 1) for i in range(n_comp)]
    print("Description SVD_DF")
    print(svd_df.head(5))

    df = pd.concat([df, svd_df], axis=1)
    return df

print("Title features ")
df = title_features(df)
print("Description features ")
df = description_features(df)

# Remove text Variables
df.drop(text,axis=1,inplace=True)
# Remove Dead Variables
df.drop(["activation_date", "image", "user_id"],axis=1,inplace=True)

# Encoder:
# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical + messy_categorical:
    df[col] = lbl.fit_transform(df[col].astype(str))
    
print("\nCatboost Modeling Stage")

X = df.loc[traindex,:].copy()
print("Training Set shape",X.shape)
test = df.loc[testdex,:].copy()
print("Submission Set Shape: {} Rows, {} Columns".format(*test.shape))
del df
gc.collect()

# Training and Validation Set
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.10, random_state=23)

# Prepare Categorical Variables
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
categorical_features_pos = column_index(X,categorical)

# Train Model
print("Train CatBoost Decision Tree")
modelstart= time.time()
cb_model = CatBoostRegressor(iterations=900,
                             learning_rate=0.08,
                             depth=10,
                             #loss_function='RMSE',
                             eval_metric='RMSE',
                             # random_seed = 23, # reminder of my mortality
                             od_type='Iter',
                             metric_period = 50,
                             task_type="GPU",
                             od_wait=20, 
                             verbose=False,
                             )

cb_model.fit(X_train, y_train,
             eval_set=(X_valid,y_valid),
             cat_features=categorical_features_pos,
             use_best_model=True, verbose=True)

print("Model Evaluation Stage")
print(cb_model.get_params())
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, cb_model.predict(X_valid))))
catpred = cb_model.predict(test)
catsub = pd.DataFrame(catpred,columns=["deal_probability"],index=testdex)
catsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
catsub.to_csv("catsub.csv",index=True,header=True)
