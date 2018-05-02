import torch
import torch.nn as nn
import utils


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AvitorEmbedding(nn.Module):
    def __init__(self, token_len, embedding_size):
        super(AvitorEmbedding, self).__init__()
        self.n_embedding_layers = len(token_len)
        self.embedding_layers = [nn.Embedding(int(tl) + 1, embedding_size) for tl in token_len]
        for i, layer in enumerate(self.embedding_layers):
            self.add_module(f"embedding_layer_{i}", layer)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        return [self.embedding_layers[i](x[:, i]) for i in range(self.n_embedding_layers)]

class Avitor(nn.Module):
    def __init__(self, embedding_model, in_features):
        super(Avitor, self).__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features=in_features, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )
        self.embedding_model = embedding_model

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        embedding_data = x[:, :self.embedding_model.n_embedding_layers].type("torch.LongTensor")
        embedding_data = utils.to_gpu(embedding_data)

        embedding_out = self.embedding_model(embedding_data)
        embedding_out = torch.cat(embedding_out, 1)

        # linear data
        linear_data = x[:, self.embedding_model.n_embedding_layers:].type("torch.FloatTensor")
        linear_data = utils.to_gpu(linear_data)

        all_features = torch.cat([embedding_out, linear_data], 1)
        all_features = utils.to_gpu(all_features)

        return self.fc(all_features)