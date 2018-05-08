import torch
import torch.utils.data as data
import numpy as np
import os
from PIL import Image


def load_img(path):
    return Image.open(path).convert("RGB")


class NumpyDataset(data.Dataset):
    def __init__(self, X, y):
        super(NumpyDataset, self).__init__()
        self.X = X
        self.Y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        X_tensor = torch.from_numpy(self.X[index,:])
        Y_tensor = torch.Tensor([self.Y[index]]) if self.Y is not None else torch.FloatTensor([0])
        return X_tensor, Y_tensor


class AvitoDataset(data.Dataset):
    def __init__(self, X_num, X_cat, X_text, y):
        super(AvitoDataset, self).__init__()
        self.X_num = X_num
        self.X_cat = X_cat
        self.X_text = X_text
        self.Y = y

    def __len__(self):
        return self.X_num.shape[0]

    def __getitem__(self, index):
        X_num_tensor = torch.from_numpy(self.X_num[index, :])
        X_cat_tensor = torch.from_numpy(self.X_cat[index, :])
        X_text_tensor = [torch.from_numpy(np.array(text[index, :].toarray())).type("torch.FloatTensor").squeeze() for
                         text in self.X_text]
        Y_tensor = torch.Tensor([self.Y[index]]) if self.Y is not None else torch.FloatTensor([0])
        return X_num_tensor, X_cat_tensor, X_text_tensor, Y_tensor


class AvitoImageDataset(data.Dataset):
    def __init__(self, root, list_img, img_size, transform):
        super(AvitoImageDataset, self).__init__()
        self.root = root
        self.list_img = list_img
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, index):
        img_name = self.list_img[index]
        img_path = f"{self.root}/{img_name}.jpg"
        if os.path.exists(img_path):
            img = load_img(img_path)
            if self.transform:
                img = self.transform(img)
        else:
            img = torch.Tensor(3, self.img_size, self.img_size)
        return img