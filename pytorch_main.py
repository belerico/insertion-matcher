import functools
import operator
from sklearn.metrics import classification_report
from dataset import get_data, BatchWrapper
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if __name__ == '__main__':
    train_ds, valid_ds, test_ds, TEXT = get_data(
        train_path='./dataset/computers/train/computers_train_medium.json',
        valid_path='./dataset/computers/valid/computers_splitted_valid_medium.json',
        test_path='./dataset/computers/test/computers_gs.json'
    )
    print(train_ds[0].title_left)
    print(valid_ds[0].title_left)
    print(test_ds[0].title_left)

    TEXT.build_vocab(train_ds, valid_ds)

    from torchtext.vocab import Vectors

    model = gensim.models.KeyedVectors.load_word2vec_format(
        './dataset/embeddings/w2v/w2v_title_300Epochs_1MinCount_9ContextWindow_100d.bin',
        binary=True)
    model.save_word2vec_format(
        fname='./dataset/embeddings/w2v/new_w2v_title_300Epochs_1MinCount_9ContextWindow_100d.bin')

    model = gensim.models.KeyedVectors.load_word2vec_format(
        './dataset/embeddings/w2v/new_w2v_title_300Epochs_1MinCount_9ContextWindow_100d.bin')
    # needed vectors not in binary form
    vectors = Vectors(name='new_w2v_title_300Epochs_1MinCount_9ContextWindow_100d.bin',
                      cache='./dataset/embeddings/w2v')  # model_name + path =
    # path_to_embeddings_file
    TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)

    from torchtext.data import BucketIterator

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train_ds, valid_ds, test_ds),
        # we pass in the datasets we want the iterator to draw data from
        batch_sizes=(32, 64, 64),
        device=torch.device('cpu'),  # if you want to use the GPU, specify the GPU number here
        sort_key=lambda x: min(max(len(x.title_left), len(x.title_right)), 20),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=True,
        shuffle=True,
        repeat=False  # we pass repeat=False because we want to wrap this Iterator layer.
    )

    train_dl = BatchWrapper(train_iter, ['title_left', 'title_right'], 'label')
    valid_dl = BatchWrapper(val_iter, ['title_left', 'title_right'], 'label')
    test_dl = BatchWrapper(test_iter, ['title_left', 'title_right'], 'label')


    class SimpleLSTMBaseline(nn.Module):
        def __init__(self, hidden_dim, emb_dim=50, conv_depth=16, kernel_size=3, pool_size=2):
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


    model = SimpleLSTMBaseline(hidden_dim=200, emb_dim=100)

    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.BCELoss()

    epochs = 15
    print("Start training")
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_corrects = 0
        for left, right, y in train_dl:  # thanks to our wrapper, we can intuitively iterate over our data!
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
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss,
                                                                                 val_loss))

    y_true = [v[2] for v in test_dl]
    y_true = functools.reduce(operator.iconcat, y_true, [])
    predictions = []
    model.eval()  # turn on evaluation mode
    for left, right, y in test_dl:
        preds = model([left, right])
        predictions.extend(preds.data > .5)
    print(classification_report(y_true, predictions))
