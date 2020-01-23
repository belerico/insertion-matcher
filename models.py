import torch
from torch import nn as nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, TEXT, hidden_dim, emb_dim=50, conv_depth=16, kernel_size=3, pool_size=2):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(TEXT.vocab.vectors)
        self.encoder_left = nn.LSTM(emb_dim, hidden_dim, num_layers=1, bidirectional=False,
                                    batch_first=True)
        self.encoder_right = nn.LSTM(emb_dim, hidden_dim, num_layers=1, bidirectional=False,
                                     batch_first=True)

        self.conv1 = nn.Conv2d(1, conv_depth, kernel_size)
        self.batch_norm1 = nn.BatchNorm2d(conv_depth)
        self.max_pool1 = nn.MaxPool2d(pool_size)
        self.mlp1 = nn.Linear(1296, 32)
        self.mlp2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq):
        hdn_left, _ = self.encoder_left(self.embedding(seq[0]))
        hdn_right, _ = self.encoder_right(self.embedding(seq[1]))
        similarity = torch.matmul(hdn_left, torch.transpose(hdn_right, 1, 2))
        similarity = torch.unsqueeze(similarity, 1)
        x = self.conv1(similarity)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.max_pool1(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(x)
        x = self.mlp1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3)
        x = self.mlp2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
