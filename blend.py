import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Any results you write to the current directory are saved as output.

root = "model_bag"

model_bag = [
    {
        "name": "xgboost",
        "path": f"{root}/xgb_tfidf0.21864_02241LB.csv",
        "score": 0.2241,
        "weight": 0.5,
    },
    {
        "name": "LightGBM",
        "path": f"{root}/lgsub_ridge_02239LB.csv",
        "score": 0.2239,
        "weight": 0.30,
    },
    # {
    #     "name": "BlendCNN",
    #     "path": f"{root}/blend_cnn.csv",
    #     "score": 0.2226,
    # },
    # {
    #     "name": "keras_birgu",
    #     "path": f"{root}/keras_keras_50k_d300_100w_avg.csv",
    #     "score": 0.2232,
    #     "weight": 0.20,
    # },
    {
        "name": "keras_cnn",
        "path": f"{root}/keras_cnn_50k_300d_100w_avg.csv",
        "score": 0.2228,
        "weight": 0.10,
    },
    {
        "name": "revert_label",
        "path": f"{root}/revert_label.csv",
        "score": 0.2232,
        "weight": 0.10,
    }
]

preds = []
for model in model_bag:
    print("\n[+] Load model {}".format(model["name"]))
    print("[+] Score {}".format(model["score"]))
    print("[+] path {}".format(model["path"]))
    label  = pd.read_csv(str(model["path"]))
    weight = model["weight"]
    preds.append(weight * label["deal_probability"].values)

preds = np.array(preds)
pred_mean = np.sum(preds, axis=0)

df = pd.DataFrame()
df['item_id'] = label['item_id']
df['deal_probability'] = pred_mean
df.to_csv(f'{root}/blend_weighted.csv',index=False)