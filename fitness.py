import functools
import operator

import torch
from sklearn.metrics import classification_report
from torch import optim as optim, nn as nn

from models import Model
import time


def fit(TEXT, train_dl, valid_dl, config, conv_depth, dense_depth, hidden_dim=100, lr=1e-3,
        loss='BCELoss'):
    model = Model(TEXT, hidden_dim=hidden_dim, conv_depth=conv_depth, dense_depth=dense_depth)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = getattr(nn, loss)()
    model.train()

    print("Start training")
    for epoch in range(1, config['epochs'] + 1):
        running_loss = 0.0
        t0 = time.time()
        for left, right, y in train_dl:
            opt.zero_grad()
            preds = model([left, right])
            loss = loss_func(preds, torch.unsqueeze(y, 1))
            loss.backward()
            opt.step()
            running_loss += loss.data.item()

        epoch_loss = running_loss / len(train_dl)

        # calculate the validation loss for this epoch
        val_loss = 0.0
        model.eval()  # turn on evaluation mode
        for left, right, y in valid_dl:
            preds = model([left, right])
            loss = loss_func(preds, torch.unsqueeze(y, 1))
            val_loss += loss.data.item()

        val_loss /= len(valid_dl)
        print('Epoch: {}, Elapsed: {:.2f}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(
            epoch, time.time() - t0,
            epoch_loss,
            val_loss))
    return model


def evaluate(model, test_dl):
    y_true = [v[2] for v in test_dl]
    y_true = functools.reduce(operator.iconcat, y_true, [])
    predictions = []
    model.eval()  # turn on evaluation mode
    for left, right, y in test_dl:
        preds = model([left, right])
        predictions.extend(preds.data > .5)
    print(classification_report(y_true, predictions))
