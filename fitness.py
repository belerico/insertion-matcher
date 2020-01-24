import functools
import operator
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from torch import optim as optim, nn as nn
from models import Model
import time


""" def fit(
    TEXT,
    train_dl,
    valid_dl,
    config,
    conv_depth,
    dense_depth,
    hidden_dim=100,
    lr=1e-3,
    kernel_size=3,
    pool_size=2,
    similarity="dot",
    loss="BCELoss",
    validate_each_epoch=True,
    trainable=False,
):
    model = Model(
        TEXT,
        hidden_dim=hidden_dim,
        conv_depth=conv_depth,
        dense_depth=dense_depth,
        similarity=similarity,
        max_len=20,
        kernel_size=kernel_size,
        pool_size=pool_size,
        trainable=trainable,
    )
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = getattr(nn, loss)()
    model.train()

    if validate_each_epoch:
        y_true = [v[2] for v in test_dl]
        y_true = functools.reduce(operator.iconcat, y_true, [])

    print("Start training")
    for epoch in range(1, config["epochs"] + 1):
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
        print(
            "Epoch: {}, Elapsed: {:.2f}s, Training Loss: {:.4f}".format(
                epoch, time.time() - t0, epoch_loss
            )
        )

        if validate_each_epoch:
            # calculate the validation loss for this epoch
            predictions = []
            val_loss = 0.0
            model.eval()  # turn on evaluation mode
            for left, right, y in valid_dl:
                preds = model([left, right])
                loss = loss_func(preds, torch.unsqueeze(y, 1))
                val_loss += loss.data.item()

            val_loss /= len(valid_dl)
            print("Validate epoch: {}, Val Loss: {:.4f}".format(epoch, val_loss))

    return model


def evaluate(model, test_dl, print_results=True):
    y_true = [v[2] for v in test_dl]
    y_true = functools.reduce(operator.iconcat, y_true, [])
    predictions = []
    model.eval()  # turn on evaluation mode
    for left, right, y in test_dl:
        preds = model([left, right])
        predictions.extend(preds.data > 0.5)

    if print_results:
        print(classification_report(y_true, predictions))
    return f1_score(y_true, predictions, average="weighted") + 1e-10 """


def fit(
    TEXT,
    train_dl,
    valid_dl,
    config,
    hidden_dim,
    conv_depth,
    kernel_size,
    dense_depth1,
    dense_depth2,
    lr=1e-3,
    pool_size=2,
    similarity="dot",
    loss="CrossEntropyLoss",
    validate_each_epoch=True,
    trainable=False,
):
    model = Model(
        TEXT,
        hidden_dim=hidden_dim,
        conv_depth=conv_depth,
        kernel_size=kernel_size,
        pool_size=pool_size,
        dense_depth1=dense_depth1,
        dense_depth2=dense_depth2,
        max_len=20,
        similarity=similarity,
        trainable=trainable,
    )
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_func = getattr(nn, loss)()
    model.train()

    if validate_each_epoch:
        y_true = [v[2] for v in valid_dl]
        y_true = functools.reduce(operator.iconcat, y_true, [])

    print("Start training")
    for epoch in range(1, config["epochs"] + 1):
        running_loss = 0.0
        t0 = time.time()
        for left, right, y in train_dl:
            opt.zero_grad()
            preds = model([left, right])
            loss = loss_func(preds, y.long())
            loss.backward()
            opt.step()
            running_loss += loss.data.item()

        epoch_loss = running_loss / len(train_dl)
        print(
            "Epoch: {}, Elapsed: {:.2f}s, Training Loss: {:.4f}".format(
                epoch, time.time() - t0, epoch_loss
            )
        )

        if validate_each_epoch:
            # calculate the validation loss for this epoch
            predictions = []
            val_loss = 0.0
            model.eval()  # turn on evaluation mode
            for left, right, y in valid_dl:
                preds = model([left, right])
                loss = loss_func(preds, y.long())
                val_loss += loss.data.item()
                predictions.extend(torch.argmax(torch.log_softmax(preds, dim=1), dim=1))

            val_loss /= len(valid_dl)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, predictions, labels=[0, 1], average="weighted"
            )
            print(
                "Validate epoch: {}, Val Loss: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}".format(
                    epoch, val_loss, prec, rec, f1
                )
            )

    return model


def evaluate(model, test_dl, print_results=True):
    y_true = [v[2] for v in test_dl]
    y_true = functools.reduce(operator.iconcat, y_true, [])
    predictions = []
    model.eval()  # turn on evaluation mode
    for left, right, y in test_dl:
        preds = model([left, right])
        predictions.extend(torch.argmax(torch.log_softmax(preds, dim=1), dim=1))

    if print_results:
        print(classification_report(y_true, predictions))
    return f1_score(y_true, predictions, average="weighted") + 1e-10
