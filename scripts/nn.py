import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')

import keras as K
from keras.layers import *
from keras.models import *
import utils

from keras.callbacks import *
from sklearn.model_selection import KFold
import argparse
import pandas as pd

from keras.callbacks import CSVLogger

from keras.utils import plot_model

# Parse the argument
def add_args(parser):
    arg = parser.add_argument
    arg('--mode', type=str, default='train', help='train mode or test mode')
    arg('--batch_size', type=int, default=1024)
    arg('--epochs', type=int, default=100)
    arg('--lr', type=float, default=5e-3)
    arg('--kfold', type=int, default=0)
    # arg('--workers', type=int, default=12)
    # arg('--device-ids', type=str, help='For example 0,1 to run on two GPUs')
    arg('--model_name', type=str, default="avito_model")

parser = argparse.ArgumentParser()
arg = parser.add_argument

add_args(parser)
args = parser.parse_args()

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
features, fnames = utils.load_features()

# Flat the features
y              = features[0]
X_num          = features[1][0]
X_cat          = features[2]
X_tfidf_text   = features[3][0]
X_tfidf_params = features[4][0]
# X_ridge_text   = features[5]
# X_ridge_params = features[6]


# RMSE function
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_model():
    input_num           = Input(shape=(X_num.shape[1],          ), name="Numeric")
    input_cat           = Input(shape=(X_cat.shape[1],          ), name="Category")
    input_tfidf_text    = Input(shape=(X_tfidf_text.shape[1],   ), name="TFIDF_text")
    input_tfidf_params  = Input(shape=(X_tfidf_params.shape[1], ), name="TFIDF_param")
    # input_ridge_text    = Input(shape=(1,                       ), name="Ridge_text")
    # input_ridge_params  = Input(shape=(1,                       ), name="Ridge_param")

    merge_input = concatenate([input_num, input_cat,
                                 input_tfidf_text, input_tfidf_params,
                                 #input_ridge_text, input_ridge_params
                            ])

    x = BatchNormalization()(merge_input)
    x = Dense(128, activation="relu", kernel_initializer="glorot_normal")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(1, activation="sigmoid", kernel_initializer="glorot_normal")(x)

    input_list = [
        input_num, input_cat,
        input_tfidf_text, input_tfidf_params,
        # input_ridge_text, input_ridge_params
    ]
    model = Model(inputs=input_list, outputs=x)
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
        skf = KFold(n_folds)
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

            X_tr_num            = X_num[train_index]
            X_tr_cat            = X_cat[train_index]
            X_tr_tfidf_text     = X_tfidf_text[train_index]
            X_tr_tfidf_params   = X_tfidf_params[train_index]
            # X_tr_ridge_text     = X_ridge_text[train_index]
            # X_tr_ridge_params   = X_ridge_params[train_index]
            y_tr                = y[train_index]

            X_va_num            = X_num[val_index]
            X_va_cat            = X_cat[val_index]
            X_va_tfidf_text     = X_tfidf_text[val_index]
            X_va_tfidf_params   = X_tfidf_params[val_index]
            # X_va_ridge_text     = X_ridge_text[val_index]
            # X_va_ridge_params   = X_ridge_params[val_index]
            y_va                = y[val_index]

            train_inputs = [
                X_tr_num, X_tr_cat,
                X_tr_tfidf_text, X_tr_tfidf_params,
                # X_tr_ridge_text, X_tr_ridge_params
            ]

            val_inputs = [
                X_va_num, X_va_cat,
                X_va_tfidf_text, X_va_tfidf_params,
                # X_va_ridge_text, X_va_ridge_params
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
            X_num, X_cat,
            X_tfidf_text, X_tfidf_params,
            # X_ridge_text, X_ridge_params
        ]

        history = model.fit(train_inputs, y, validation_split=0.1,
                            verbose=1, callbacks=callbacks_list,
                            epochs=args.epochs, batch_size=args.batch_size)

def test():
    predict_root = "predict"
    checkpoint_path = f"checkpoint/{args.model_name}/"
    file_path = f"{checkpoint_path}/{args.model_name}_best.h5"
    sample_submission = "/User/ngxbac/project/kaggle/avito/sample_submission.csv"

    n_folds = args.kfold

    test_inputs = [
        X_num, X_cat,
        X_tfidf_text, X_tfidf_params,
        # X_ridge_text, X_ridge_params
    ]

    if n_folds:
        # Test with k-fold
        preds_all = []
        for fold in range(n_folds):
            model = get_model()
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