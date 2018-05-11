import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings
warnings.filterwarnings('ignore')

import keras as K
from keras.layers import *
from keras.models import *
import json
import utils
from scipy.sparse import hstack
from keras.callbacks import *
import gc
from sklearn.model_selection import train_test_split


from keras.utils import plot_model
config = json.load(open("config.json"))

# Load json config
config = json.load(open("config.json"))
extracted_features_root = config["extracted_features"]
# Load data and token len of embedding layers
print("[+] Load features ...")
y = utils.load_features(extracted_features_root, "y_train")
token_len = utils.load_features(extracted_features_root, "token_len")

X_train_num = utils.load_features(extracted_features_root, "X_train_num")
X_train_cat = utils.load_features(extracted_features_root, "X_train_cat")
X_train_desc = utils.load_features(extracted_features_root, "X_train_desc").any()
X_train_title = utils.load_features(extracted_features_root, "X_train_title").any()
X_train_text = [X_train_desc, X_train_title]

del X_train_desc, X_train_title
gc.collect()

cat_columns = [
    "region", # Importance feature, best_val: 0.507
    "city", # Importance feature, best_val: 0.510
    "parent_category_name", # This feature seems not be importance, best_val:0.0505
    "category_name", # This feature seems be importance, best_val:0.0507
    "param_1", "param_2", "param_3",
    "user_type", "image_top_1"
]

text_columns = [
    "description",
    "title"
]


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def get_model():
    input_num = Input(shape=(X_train_num.shape[1],), name="Numeric")

    input_cat = []
    for cat, name in zip(range(X_train_cat.shape[1]), cat_columns):
        input_cat.append(Input(shape=(1,), name="cat_" + name))

    input_text = []
    for text, name in zip(X_train_text, text_columns):
        input_text.append(Input(shape=(text.shape[1], ), name="text_"+name))

    out_num = BatchNormalization()(input_num)
    out_num = Dense(50, activation="relu", kernel_initializer="glorot_normal")(out_num)
    out_num = BatchNormalization()(out_num)
    out_num = Dropout(0.5)(out_num)

    out_cat = []
    for x, tkl in zip(input_cat, token_len):
        x = Embedding(tkl + 1, config["embedding_size"], embeddings_initializer ="glorot_normal")(x)
        x = Flatten()(x)
        out_cat.append(x)

    out_text = []
    text_input_size = [txt.shape[1] for txt in X_train_text]
    for x, text_size in zip(input_text, text_input_size):
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(text_size // 100, activation="linear", kernel_initializer="glorot_normal")(x)
        x = Dropout(0.5)(x)
        out_text.append(x)

    merg_out_text = concatenate(out_text)
    merg_out_text = BatchNormalization()(merg_out_text)
    merg_out_text = Dense(128, activation="relu", kernel_initializer="glorot_normal")(merg_out_text)
    merg_out_text = Dropout(0.5)(merg_out_text)

    merge_out = concatenate([out_num, *out_cat, merg_out_text])
    merge_out = BatchNormalization()(merge_out)
    merge_out = Dense(50, activation="relu", kernel_initializer="glorot_normal")(merge_out)
    merge_out = BatchNormalization()(merge_out)
    merge_out = Dense(1, activation="sigmoid", kernel_initializer="glorot_normal")(merge_out)

    model = Model(inputs=[input_num, *input_cat, *input_text], outputs=merge_out)
    model.compile(optimizer=optimizers.Adam(lr=config["lr"]), loss="mean_squared_error", metrics=[root_mean_squared_error])
    return model


if __name__ == '__main__':
    model = get_model()
    model.summary()
    plot_model(model, to_file='model.png')

    file_path = 'simpleRNN3.h5'
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=True,
                                 mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=4)
    lr_reduced = ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=2,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='min')
    callbacks_list = [checkpoint, early, lr_reduced]

    list_train_cat = [X_train_cat[:, i] for i in range(X_train_cat.shape[1])]

    history = model.fit([X_train_num, *list_train_cat, *X_train_text], y, validation_split=0.1,
                        verbose=1, callbacks=callbacks_list,
                        epochs=10, batch_size=512)


