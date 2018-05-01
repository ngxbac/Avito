import torch
import torch.utils.data as data


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