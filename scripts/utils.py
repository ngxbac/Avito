import bcolz
import os


def save_bcolz(feature, root, name):
    if not os.path.exists(root):
        os.mkdir(root)
    bcolz.carray(feature, chunklen=1024, mode='w', rootdir=f"{root}/{name}")

def load_bcolz(root, name):
    if not os.path.exists(root):
        print(f"[+] Feature {name} does not exists")
        return None
    return bcolz.open(f"{root}/{name}")

def save_csv(df, root, name):
    if not os.path.exists(root):
        os.mkdir(root)
    df.to_csv(f"{root}/{name}", index=False)

def load_features(train=True):
    # Load numeric features
    root = "features"

    if train:
        prefix = "train"
    else:
        prefix = "test"

    ato_prefix = [
        "y", "num", "cat",
        "tfidf_text", "tfidf_params",
        "ridge_text", "ridge_params"
    ]

    fname = [f"X_{prefix}_{x}" for x in ato_prefix]
    features = [load_bcolz(root, name) for name in fname]

    return features, fname