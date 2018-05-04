import json
import pandas as pd
import bcolz
import numpy as np
import os
from tqdm import tqdm
from sys import getsizeof
from nltk.corpus import stopwords
import utils

stopWords_rus = stopwords.words('russian')

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
    "region", # Importance feature, best_val: 0.507
    "city", # Importance feature, best_val: 0.510
    "parent_category_name", # This feature seems not be importance, best_val:0.0505
    "category_name", # This feature seems be importance, best_val:0.0507
    "param_1", "param_2", "param_3",
    "user_type", "image_top_1"
]

agg_columns = [
    'region', 'city', 'parent_category_name',
    'category_name', 'image_top_1', 'user_type',
    'item_seq_number','activation_date'
]

extract_columns = ["item_seq_number"]

config = json.load(open("config.json"))


def load_csv(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["activation_date"])
    df = df.replace(np.nan, -1, regex=True)
    return df


def remove_unused_columns(df: pd.DataFrame, columns=unused_columns):
    for c in columns:
        df = df.drop(c, axis=1)
    return df


def date_to_dow(df: pd.DataFrame):
    df["weekday"] = df["activation_date"].dt.weekday
    df["week"] = df["activation_date"].dt.week
    df["mon"] = df["activation_date"].dt.month
    df["mday"] = df["activation_date"].dt.day
    df["year_day"] = df["activation_date"].dt.dayofyear

    extract_columns.append("weekday")
    # extract_columns.append("week")
    # extract_columns.append("mon")
    # extract_columns.append("mday")
    # extract_columns.append("year_day")

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


def agg_features(train_df, test_df, columns=agg_columns):
    # for c in columns:
    #     gp = train_df.groupby(c)['deal_probability']
    #     mean = gp.mean()
    #     std = gp.std()
    #     train_df[c + '_deal_probability_avg'] = train_df[c].map(mean)
    #     train_df[c + '_deal_probability_std'] = train_df[c].map(std)
    #
    #     test_df[c + '_deal_probability_avg'] = test_df[c].map(mean)
    #     test_df[c + '_deal_probability_std'] = test_df[c].map(std)
    #
    #     extract_columns.append(c + '_deal_probability_avg')
    #     extract_columns.append(c + '_deal_probability_std')

    for c in columns:
        gp = train_df.groupby(c)['price']
        mean = gp.mean()
        train_df[c + '_price_avg'] = train_df[c].map(mean)
        test_df[c + '_price_avg'] = test_df[c].map(mean)

        extract_columns.append(c + '_price_avg')

    return train_df, test_df


def getTextFeatures(T, Col, max_features=10000, ngrams=(1,2), verbose=True):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD
    from nltk.stem import PorterStemmer
    import re
    p = PorterStemmer()

    def wordPreProcess(sentence):
        return ' '.join([p.stem(x.lower()) for x in re.split('\W', sentence) if len(x) >= 1])

    if verbose:
        print('processing: ', Col)
    vectorizer = CountVectorizer(stop_words=stopWords_rus,
                                 preprocessor=wordPreProcess,
                                 max_features=max_features,
                                 binary=True,
                                 ngram_range=ngrams)
#     vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'),
#                                  preprocessor=wordPreProcess,
#                                  max_features=max_features)
    X = vectorizer.fit_transform(T[Col]).toarray()
    return X, vectorizer.get_feature_names()


def title_features(train_df, test_df):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD

    train_df["title_nwords"] = train_df["title"].apply(lambda x: len(x.split()))
    test_df["title_nwords"] = test_df["title"].apply(lambda x: len(x.split()))
    extract_columns.append("title_nwords")

    tfidf_vec = TfidfVectorizer(ngram_range=(1, 1))
    full_tfidf = tfidf_vec.fit_transform(train_df['title'].values.tolist() + test_df['title'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['title'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['title'].values.tolist())

    ### SVD Components ###
    n_comp = 3
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    train_svd.columns = ['svd_title_' + str(i + 1) for i in range(n_comp)]
    test_svd.columns = ['svd_title_' + str(i + 1) for i in range(n_comp)]
    for i in range(n_comp):
        extract_columns.append('svd_title_' + str(i + 1))

    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)

    return train_df, test_df


def description_features(train_df, test_df):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD

    ## Filling missing values ##
    train_df["description"].fillna("NA", inplace=True)
    test_df["description"].fillna("NA", inplace=True)

    train_df["description"] = train_df["description"].apply(lambda x: str(x))
    test_df["description"] = test_df["description"].apply(lambda x: str(x))

    train_df["desc_nwords"] = train_df["description"].apply(lambda x: len(str(x).split()))
    test_df["desc_nwords"] = test_df["description"].apply(lambda x: len(str(x).split()))
    extract_columns.append("desc_nwords")

    ### TFIDF Vectorizer ###
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 1), max_features=100000)
    full_tfidf = tfidf_vec.fit_transform(
        train_df['description'].values.tolist() + test_df['description'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['description'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['description'].values.tolist())

    ### SVD Components ###
    n_comp = 3
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    train_svd.columns = ['svd_desc_' + str(i + 1) for i in range(n_comp)]
    test_svd.columns = ['svd_desc_' + str(i + 1) for i in range(n_comp)]
    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)
    for i in range(n_comp):
        extract_columns.append('svd_desc_' + str(i + 1))

    return train_df, test_df


def extract_title_description_features(train_df, test_df):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD

    tfidf = TfidfVectorizer(max_features=50000, stop_words=stopWords_rus)
    tfidf_title = TfidfVectorizer(max_features=50000, stop_words=stopWords_rus)

    train_df['description'] = train_df['description'].fillna(' ')
    test_df['description'] = test_df['description'].fillna(' ')
    train_df['title'] = train_df['title'].fillna(' ')
    test_df['title'] = test_df['title'].fillna(' ')

    train_df["description"] = train_df["description"].apply(lambda x: str(x))
    test_df["description"] = test_df["description"].apply(lambda x: str(x))

    train_df["title_nwords"] = train_df["title"].apply(lambda x: len(x.split()))
    test_df["title_nwords"] = test_df["title"].apply(lambda x: len(x.split()))
    extract_columns.append("title_nwords")

    train_df["desc_nwords"] = train_df["description"].apply(lambda x: len(x.split()))
    test_df["desc_nwords"] = test_df["description"].apply(lambda x: len(x.split()))
    extract_columns.append("desc_nwords")


    tfidf.fit(pd.concat([train_df['description'], test_df['description']]))
    tfidf_title.fit(pd.concat([train_df['title'], test_df['title']]))

    train_des_tfidf = tfidf.transform(train_df['description'])
    test_des_tfidf = tfidf.transform(test_df['description'])

    train_title_tfidf = tfidf.transform(train_df['title'])
    test_title_tfidf = tfidf.transform(test_df['title'])

    n_comp = 3
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(tfidf.transform(pd.concat([train_df['description'], test_df['description']])))

    svd_title = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_title.fit(tfidf.transform(pd.concat([train_df['title'], test_df['title']])))

    train_svd = pd.DataFrame(svd_obj.transform(train_des_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_des_tfidf))
    train_svd.columns = ['svd_des_' + str(i + 1) for i in range(n_comp)]
    test_svd.columns = ['svd_des_' + str(i + 1) for i in range(n_comp)]
    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)

    train_title_svd = pd.DataFrame(svd_title.transform(train_title_tfidf))
    test_titile_svd = pd.DataFrame(svd_title.transform(test_title_tfidf))
    train_title_svd.columns = ['svd_title_' + str(i + 1) for i in range(n_comp)]
    test_titile_svd.columns = ['svd_title_' + str(i + 1) for i in range(n_comp)]
    train_df = pd.concat([train_df, train_title_svd], axis=1)
    test_df = pd.concat([test_df, test_titile_svd], axis=1)

    for i in range(n_comp):
        extract_columns.append('svd_des_' + str(i + 1))
        extract_columns.append('svd_title_' + str(i + 1))

    return train_df, test_df


def extract_params_features(train_df, test_df):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import TruncatedSVD

    train_df["param_1"].fillna("NA", inplace=True)
    test_df["param_1"].fillna("NA", inplace=True)
    train_df["param_1"] = train_df["param_1"].apply(lambda x: str(x))
    test_df["param_1"] = test_df["param_1"].apply(lambda x: str(x))

    train_df["param_2"].fillna("NA", inplace=True)
    test_df["param_2"].fillna("NA", inplace=True)
    train_df["param_2"] = train_df["param_2"].apply(lambda x: str(x))
    test_df["param_2"] = test_df["param_2"].apply(lambda x: str(x))

    train_df["param_3"].fillna("NA", inplace=True)
    test_df["param_3"].fillna("NA", inplace=True)
    train_df["param_3"] = train_df["param_3"].apply(lambda x: str(x))
    test_df["param_3"] = test_df["param_3"].apply(lambda x: str(x))


    train_df["params"] = train_df["param_1"] + train_df["param_2"] + train_df["param_3"]
    test_df["params"] = test_df["param_1"] + test_df["param_2"] + test_df["param_3"]

    train_df["param_nwords"] = train_df["params"].apply(lambda x: len(str(x).split()))
    test_df["param_nwords"] = test_df["params"].apply(lambda x: len(str(x).split()))
    extract_columns.append("param_nwords")

    ### TFIDF Vectorizer ###
    tfidf_vec = TfidfVectorizer(ngram_range=(1, 1), max_features=100000)
    full_tfidf = tfidf_vec.fit_transform(
        train_df['params'].values.tolist() + test_df['params'].values.tolist())
    train_tfidf = tfidf_vec.transform(train_df['params'].values.tolist())
    test_tfidf = tfidf_vec.transform(test_df['params'].values.tolist())

    ### SVD Components ###
    n_comp = 3
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    train_svd.columns = ['svd_param_' + str(i + 1) for i in range(n_comp)]
    test_svd.columns = ['svd_param_' + str(i + 1) for i in range(n_comp)]
    train_df = pd.concat([train_df, train_svd], axis=1)
    test_df = pd.concat([test_df, test_svd], axis=1)
    for i in range(n_comp):
        extract_columns.append('svd_param_' + str(i + 1))

    return train_df, test_df

from scipy.sparse import hstack
def main():
    # Extract train data
    print("[+] Extract train data ")
    print("[+] Read csv file...")
    train_df = load_csv(config["train_csv"])
    test_df = load_csv(config["test_csv"])

    print("[+] Remove unused columns ...")
    train_df = remove_unused_columns(train_df)
    test_df = remove_unused_columns(test_df)

    print("[+] Convert date to day of week ...")
    train_df = date_to_dow(train_df)
    test_df = date_to_dow(test_df)

    # print("Agg data ...")
    # train_df, test_df = agg_features(train_df, test_df)

    print("[+] Extract title features ...")
    train_df, test_df = title_features(train_df, test_df)

    print("[+] Extract description features ...")
    train_df, test_df = description_features(train_df, test_df)

    # print("Extract param features ...")
    # train_df, test_df = extract_params_features(train_df, test_df)

    # print("Extract title and description features ...")
    # train_df, test_df = extract_title_description_features(train_df, test_df)

    print("[+] Create token ...")
    token = create_token(train_df)

    print("[+] Tokenize data ...")
    X_train = tokenize_data(train_df, token)
    X_test = tokenize_data(test_df, token)

    print("[+] Make matrix data ...")
    X_train.append(log_prices(train_df))
    X_test.append(log_prices(test_df))

    for c in extract_columns:
        X_train.append(train_df[c].as_matrix())
        X_test.append(test_df[c].as_matrix())

    print("[+] Save features ...")
    y_train = train_df["deal_probability"].as_matrix()
    X_train = np.asarray(X_train).T
    X_test = np.asarray(X_test).T
    y_train = np.asarray(y_train)
    # Save token len
    token_len = [len(t) for t in token]
    
    extracted_features_root = config["extracted_features"]
    utils.save_features(X_train, root=extracted_features_root,
                       name="X_train")
    utils.save_features(X_test, root=extracted_features_root,
                       name="X_test")
    utils.save_features(y_train, root=extracted_features_root,
                       name="y_train")
    utils.save_features(np.asarray(token_len), 
                        root=extracted_features_root,
                        name="token_len")


if __name__ == '__main__':
    main()