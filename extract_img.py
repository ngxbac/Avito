import torch
import torchvision.models as torch_model
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import json
import os
import utils
from tqdm import tqdm
import pandas as pd
import numpy as np
import bcolz
import datasets


N_FEATURES = 1000
IMG_SIZE = 224
BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
])

def get_model():
    return torch_model.vgg16(pretrained=True)


def create_empty_bcolz(n, name):
    return bcolz.carray(np.zeros((0,n), np.float32), chunklen=1, mode='w', rootdir=name)


def main():
    # Load config file
    config = json.load(open("config.json"))
    # Load train and test csv
    train_df = pd.read_csv(config["train_csv"], usecols=["image"])
    test_df = pd.read_csv(config["test_csv"], usecols=["image"])

    # Export to list
    train_list = train_df["image"].tolist()
    test_list = test_df["image"].tolist()
    # Remove
    del train_df, test_df


    roots = [
        "train_root",
        "test_root"
    ]
    list_prefix = ["train", "test"]
    lists = [train_list, test_list]

    model = get_model()
    if torch.cuda.is_available():
        model.cuda()

    for prefix, root, img_list in zip(list_prefix, roots, lists):
        print(f"[+] Extract {prefix} data")
        dataset = datasets.AvitoImageDataset(root, img_list, IMG_SIZE, transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        bcolz_arr = create_empty_bcolz(N_FEATURES, f"X_{prefix}_img")

        model.eval()
        for data in tqdm(dataloader, total=len(dataloader)):
            data = utils.to_gpu(data)
            features = model(data)
            bcolz_arr.append(features.data.cpu().numpy())
            bcolz_arr.flush()


if __name__ == '__main__':
    main()