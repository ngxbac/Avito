import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')

import keras as K
from keras.layers import *
from keras.models import *
from keras.regularizers import l2, l1
import utils

from keras.callbacks import *
from sklearn.model_selection import KFold
import argparse
import pandas as pd
import keras_model as kmodel

from keras.callbacks import CSVLogger

from keras.utils import plot_model

# Parse the argument
def add_args(parser):
    arg = parser.add_argument
    arg('--mode', type=str, default='train', help='train mode or test mode')
    arg('--batch_size', type=int, default=1024)
    arg('--epochs', type=int, default=100)
    arg('--lr', type=float, default=5e-3)
    arg('--kfold', type=int, default=10)
    # arg('--workers', type=int, default=12)
    # arg('--device-ids', type=str, help='For example 0,1 to run on two GPUs')
    arg('--model_name', type=str, default="new_nn")

parser = argparse.ArgumentParser()
arg = parser.add_argument

add_args(parser)
args = parser.parse_args()

if args.mode == "train":
    is_train = True
else:
    is_train = False

# Load all the features
"""
The order of features should be
- y
- num
- cat
- tfidf_text
- tfidf_params
- ridge_text
- ridge_params
"""
features, fnames = utils.load_features(is_train)

# Flat the features
y              = features[0]
X_num          = features[1][0]
X_cat          = features[2]
# X_tfidf_text   = features[3][0]
# X_tfidf_params = features[4][0]
# X_ridge_text   = features[5]
# X_ridge_params = features[6]
X_word         = features[7]
embedding_weights = features[8]

if is_train:
    cat_token_len = []
    for cat in range(X_cat.shape[1]):
        tmp = X_cat[:, cat]
        cat_token_len.append(len(np.unique(tmp)))

    np.save("features/token_len.npy", np.array(cat_token_len))
else:
    cat_token_len = np.load("features/token_len.npy")

# Define the list that unused
unused_num = [
]

use_num = [
    "image_top_1",
    "item_seq_number",
    "price",
    "description_num_words",
    "title_num_words"
]

X_num_num = utils.use_numeric(X_num, use_num)

use_region_st = [
    "region_dp_mean",
    "region_dp_std",
    "region_price_mean",
    "region_price_std",
    "region_to_price",
]

X_region_st = utils.use_numeric(X_num, use_region_st)


use_city_st = [
    "city_dp_mean",
    "city_dp_std",
    "city_price_mean",
    "city_price_std",
    "city_to_price",
]

X_city_st = utils.use_numeric(X_num, use_city_st)

use_parent_cat_st = [
    "parent_category_name_dp_mean",
    "parent_category_name_dp_std",
    "parent_category_name_price_mean",
    "parent_category_name_price_std",
    "parent_category_name_to_price",
]

X_parent_cat_st = utils.use_numeric(X_num, use_parent_cat_st)

use_cat_name_st = [
    "category_name_dp_mean",
    "category_name_dp_std",
    "category_name_price_mean",
    "category_name_price_std",
    "category_name_to_price",
]

X_cat_name_st = utils.use_numeric(X_num, use_cat_name_st)


unused_cat = [
    # "weekday",
    # "param_3"
    # "ads_count"
    "no_p1",
    "no_p2",
    "no_p3"
]

X_cat = utils.unused_category(X_cat, unused_cat)
# print("[+] Cat features \n{}".format(cat_keep_list))

print(X_num_num.shape)
print(X_region_st.shape)
print(X_city_st.shape)
print(X_parent_cat_st.shape)
print(X_cat_name_st.shape)
print(X_cat.shape)


# RMSE function
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_model():
    input_num           = Input(shape=(X_num_num.shape[1],), name="Numeric")
    input_reg_st        = Input(shape=(X_region_st.shape[1],), name="Region_st")
    input_city_st       = Input(shape=(X_city_st.shape[1],), name="city_st")
    input_parent_cat_st = Input(shape=(X_parent_cat_st.shape[1],), name="parent_st")
    input_cat_name_st   = Input(shape=(X_cat_name_st.shape[1],), name="cat_name_st")
    input_cat           = Input(shape=(X_cat.shape[1],), name="Category")
    input_words         = Input((100,), name="word")

    x_num = BatchNormalization()(input_num)
    x_num = Dense(32, activation="relu", kernel_initializer="glorot_normal")(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.25)(x_num)

    x_reg_st = BatchNormalization()(input_reg_st)
    x_reg_st = Dense(16, activation="relu", kernel_initializer="glorot_normal")(x_reg_st)
    x_reg_st = BatchNormalization()(x_reg_st)
    x_reg_st = Dropout(0.25)(x_reg_st)

    x_city_st = BatchNormalization()(input_city_st)
    x_city_st = Dense(16, activation="relu", kernel_initializer="glorot_normal")(x_city_st)
    x_city_st = BatchNormalization()(x_city_st)
    x_city_st = Dropout(0.25)(x_city_st)

    x_parent_cat_st = BatchNormalization()(input_parent_cat_st)
    x_parent_cat_st = Dense(16, activation="relu", kernel_initializer="glorot_normal")(x_parent_cat_st)
    x_parent_cat_st = BatchNormalization()(x_parent_cat_st)
    x_parent_cat_st = Dropout(0.25)(x_parent_cat_st)

    x_cat_name_st = BatchNormalization()(input_cat_name_st)
    x_cat_name_st = Dense(16, activation="relu", kernel_initializer="glorot_normal")(x_cat_name_st)
    x_cat_name_st = BatchNormalization()(x_cat_name_st)
    x_cat_name_st = Dropout(0.25)(x_cat_name_st)

    x_num = concatenate([x_num, x_reg_st, x_city_st, x_parent_cat_st, x_cat_name_st])

    cat_embeds = []
    for idx in range(X_cat.shape[1]):
        x_cat = Lambda(lambda x: x[:, idx, None])(input_cat)
        x_cat = Embedding(cat_token_len[idx] + 1, 4, input_length=1)(x_cat)
        # x_cat = SpatialDropout1D(0.25)(x_cat)
        x_cat = Flatten()(x_cat)
        cat_embeds.append(x_cat)

    embeds = concatenate(cat_embeds)
    embeds = BatchNormalization()(embeds)
    embeds = Dropout(0.2)(embeds)

    x_words = kmodel.CapsuleNet(input_words, 50000, 300, embedding_weights)
    x_words = BatchNormalization()(x_words)
    x_words = Dropout(0.2)(x_words)

    x = concatenate([x_num, embeds, x_words])
    x = BatchNormalization()(x)
    x = Dense(32, activation="relu", kernel_initializer="glorot_normal")(x)
    x = BatchNormalization()(x)
    outp = Dense(1, activation="sigmoid", kernel_initializer="glorot_normal")(x)

    input_list = [
        input_num,
        input_reg_st,
        input_city_st,
        input_parent_cat_st,
        input_cat_name_st,
        input_cat,
        input_words
    ]
    model = Model(inputs=input_list, outputs=outp)
    model.compile(optimizer=optimizers.Adam(lr=args.lr), loss="mean_squared_error", metrics=[rmse])
    return model


def train():
    checkpoint_path = f"checkpoint/{args.model_name}/"

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    file_path = f"{checkpoint_path}/{args.model_name}_best.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2,
                                 save_best_only=True, save_weights_only=True,
                                 mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=1e-4)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=3,
                                   verbose=1,
                                   epsilon=1e-4,
                                   min_lr=1e-5,
                                   mode='min')

    n_folds = args.kfold

    if n_folds:
        # Train with k-fold
        skf = KFold(n_folds, shuffle=True, random_state=2018)
        for fold, (train_index, val_index) in enumerate(skf.split(X_num)):

            model = get_model()
            # model.summary()

            print(f"\n[+] Fold {fold}")

            file_path = f"{checkpoint_path}/{args.model_name}_best_{fold}.h5"
            checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True,
                                         save_weights_only=True,
                                         mode='min')
            csv_logger = CSVLogger(f'{checkpoint_path}/log_{args.model_name}_{fold}.csv', append=True, separator=',')
            callbacks_list = [checkpoint, early, lr_reduced, csv_logger]


            X_tr_num            = X_num_num[train_index]
            X_tr_cat            = X_cat[train_index]
            X_tr_reg_st         = X_region_st[train_index]
            X_tr_city_st        = X_city_st[train_index]
            X_tr_parent_cat_st  = X_parent_cat_st[train_index]
            X_tr_cat_name_st    = X_cat_name_st[train_index]
            X_tr_word              = X_word[train_index]
            y_tr                = y[train_index]

            X_va_num            = X_num_num[val_index]
            X_va_cat            = X_cat[val_index]
            X_va_reg_st         = X_region_st[val_index]
            X_va_city_st        = X_city_st[val_index]
            X_va_parent_cat_st  = X_parent_cat_st[val_index]
            X_va_cat_name_st    = X_cat_name_st[val_index]
            X_va_word           = X_word[val_index]
            y_va                = y[val_index]

            train_inputs = [
                X_tr_num,
                X_tr_reg_st,
                X_tr_city_st,
                X_tr_parent_cat_st,
                X_tr_cat_name_st,
                X_tr_cat,
                X_tr_word
            ]

            val_inputs = [
                X_va_num,
                X_va_reg_st,
                X_va_city_st,
                X_va_parent_cat_st,
                X_va_cat_name_st,
                X_va_cat,
                X_va_word
            ]

            history = model.fit(train_inputs, y_tr,
                                validation_data=(val_inputs, y_va),
                                verbose=1, callbacks=callbacks_list,
                                epochs=args.epochs, batch_size=args.batch_size)
    else:
        model = get_model()
        # model.summary()

        csv_logger = CSVLogger(f'{checkpoint_path}/log_{args.model_name}_one.csv', append=True, separator=',')
        callbacks_list = [checkpoint, early, lr_reduced, csv_logger]

        train_inputs = [
            X_num_num,
            X_region_st,
            X_city_st,
            X_parent_cat_st,
            X_cat_name_st,
            X_cat,
            X_word
        ]

        history = model.fit(train_inputs, y, validation_split=0.1,
                            verbose=1, callbacks=callbacks_list,
                            epochs=args.epochs, batch_size=args.batch_size)

def test():
    predict_root = "predict"
    checkpoint_path = f"checkpoint/{args.model_name}/"
    file_path = f"{checkpoint_path}/{args.model_name}_best.h5"
    sample_submission = "/home/deeplearning/Kaggle/avito/input/sample_submission.csv"

    n_folds = args.kfold

    test_inputs = [
        X_num_num,
        X_region_st,
        X_city_st,
        X_parent_cat_st,
        X_cat_name_st,
        X_cat,
        X_word
    ]

    if n_folds:
        # Test with k-fold
        preds_all = []
        for fold in range(n_folds):
            model = get_model()
            model.summary()
            print(f"\n[+] Test Fold {fold}")
            file_path = f"{checkpoint_path}/{args.model_name}_best_{fold}.h5"
            model.load_weights(file_path)
            pred = model.predict(test_inputs, batch_size=args.batch_size)
            submission = pd.read_csv(sample_submission)
            submission['deal_probability'] = pred
            utils.save_csv(submission, predict_root, f"{args.model_name}_{fold}.csv")
            preds_all.append(pred)
        preds_all = np.array(preds_all)
        preds_avg = np.mean(preds_all, axis=0)
        submission = pd.read_csv(sample_submission)
        submission['deal_probability'] = preds_avg
        # submission.to_csv(f"submission_{model_name}_avg.csv", index=False)
        utils.save_csv(submission, predict_root, f"{args.model_name}_avg.csv")
    else:
        model = get_model()
        model.summary()
        model.load_weights(file_path)
        pred = model.predict(test_inputs, batch_size=1024)
        submission = pd.read_csv(sample_submission)
        submission['deal_probability'] = pred
        utils.save_csv(submission, predict_root, f"keras_{args.model_name}_one.csv")


if __name__ == '__main__':
    mode = args.mode
    if mode == "train":
        train()
    else:
        test()