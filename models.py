import torch
from torch import nn as nn
from torch.nn import functional as F


""" class Model(nn.Module):
    def __init__(
        self,
        TEXT,
        hidden_dim,
        conv_depth,
        kernel_size,
        pool_size,
        dense_depth,
        max_len,
        similarity,
        trainable,
    ):
        super().__init__()
        embedding_matrix = TEXT.vocab.vectors
        emb_dim = embedding_matrix.size()[1]

        self.similarity = similarity
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding.requires_grad = trainable

        self.encoder_left = nn.LSTM(
            emb_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True
        )
        self.encoder_right = nn.LSTM(
            emb_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True
        )

        self.conv1 = nn.Conv2d(1, conv_depth, kernel_size)
        self.batch_norm1 = nn.BatchNorm2d(conv_depth)
        output_size = int((((max_len - 2) / 2) ** 2) * conv_depth)
        self.max_pool1 = nn.MaxPool2d(pool_size)
        self.mlp1 = nn.Linear(output_size, dense_depth)
        self.mlp2 = nn.Linear(dense_depth, 16)
        self.out = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq):
        hdn_left, _ = self.encoder_left(self.embedding(seq[0]))
        hdn_right, _ = self.encoder_right(self.embedding(seq[1]))

        if self.similarity == "dot":
            similarity = torch.matmul(hdn_left, torch.transpose(hdn_right, 1, 2))
            similarity = torch.unsqueeze(similarity, 1)
        else:
            num = torch.matmul(hdn_left, torch.transpose(hdn_right, 1, 2))
            n1 = torch.norm(hdn_left, dim=-1)
            n2 = torch.norm(hdn_right, dim=-1)
            den_for_row = (n1 * n2)[:, :, None]
            repeat_along_last_axis = num.size()[-1]
            similarity = num * torch.repeat_interleave(
                den_for_row, repeat_along_last_axis, dim=-1
            )
            similarity = torch.unsqueeze(similarity, 1)

        x = self.conv1(similarity)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.max_pool1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3)
        x = self.mlp2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3)
        x = self.out(x)
        x = self.sigmoid(x)
        return x """

class Model(nn.Module):
    def __init__(
        self,
        TEXT,
        hidden_dim,
        conv_depth,
        kernel_size,
        pool_size,
        dense_depth,
        max_len,
        similarity,
        trainable,
    ):
        super().__init__()
        embedding_matrix = TEXT.vocab.vectors
        emb_dim = embedding_matrix.size()[1]

        self.similarity = similarity
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.embedding.requires_grad = trainable

        self.encoder_left = nn.LSTM(
            emb_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True
        )
        self.encoder_right = nn.LSTM(
            emb_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True
        )

        self.conv1 = nn.Conv2d(1, conv_depth, kernel_size)
        self.batch_norm1 = nn.BatchNorm2d(conv_depth)
        output_size = int((((max_len - 2) / 2) ** 2) * conv_depth)
        self.max_pool1 = nn.MaxPool2d(pool_size)
        self.mlp1 = nn.Linear(output_size, dense_depth)
        self.mlp2 = nn.Linear(dense_depth, 16)
        self.out = nn.Linear(16, 2)

    def forward(self, seq):
        hdn_left, _ = self.encoder_left(self.embedding(seq[0]))
        hdn_right, _ = self.encoder_right(self.embedding(seq[1]))

        if self.similarity == "dot":
            similarity = torch.matmul(hdn_left, torch.transpose(hdn_right, 1, 2))
            similarity = torch.unsqueeze(similarity, 1)
        else:
            num = torch.matmul(hdn_left, torch.transpose(hdn_right, 1, 2))
            n1 = torch.norm(hdn_left, dim=-1)
            n2 = torch.norm(hdn_right, dim=-1)
            den_for_row = (n1 * n2)[:, :, None]
            repeat_along_last_axis = num.size()[-1]
            similarity = num * torch.repeat_interleave(
                den_for_row, repeat_along_last_axis, dim=-1
            )
            similarity = torch.unsqueeze(similarity, 1)

        x = self.conv1(similarity)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.max_pool1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.mlp1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3)
        x = self.mlp2(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3)
        x = self.out(x)
        return x
