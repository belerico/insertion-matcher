from keras.utils import Sequence
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from utils import parse_content_line


class Dataset:
    def __init__(self, file_txt, num_words=1000, max_len=20):
        contents = np.concatenate([parse_content_line(x) for x in open(file_txt, "r").readlines()],
                                  axis=0)
        self.tokenizer = Tokenizer(num_words=num_words)
        self.tokenizer.fit_on_texts(contents[:, 0])
        self.tokenizer.fit_on_texts(contents[:, 1])
        self.dataset = np.zeros([len(contents), 2, max_len])
        self.labels = contents[:, 2]

        self.dataset[:, 0, :] = pad_sequences(self.tokenizer.texts_to_sequences(contents[:, 0]), max_len)
        self.dataset[:, 1, :] = pad_sequences(self.tokenizer.texts_to_sequences(contents[:, 1]), max_len)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        item = self.dataset[key]
        first = item[0]
        second = item[1]
        label = np.array([int(self.labels[key])])
        return first, second, label


class Dataloader(Sequence):
    def __init__(self, dataset, batch_size=32, indexes: np.ndarray = None):
        self.dataset = dataset
        self.indexes = indexes if indexes is not None else np.arange(len(self.dataset))
        self.batch_size = batch_size

    def __getitem__(self, key):
        ids = key
        idf = key + self.batch_size
        first_titles = []
        second_titles = []
        labels = []

        for id_item in range(ids, idf):
            first, second, label = self.dataset[self.indexes[id_item]]
            first = first[np.newaxis, :]
            second = second[np.newaxis, :]
            label = label[np.newaxis, :]

            first_titles.append(first)
            second_titles.append(second)
            labels.append(label)

        first_titles = np.concatenate(first_titles, axis=0)
        second_titles = np.concatenate(second_titles, axis=0)
        labels = np.concatenate(labels, axis=0)

        return [first_titles, second_titles], labels

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.indexes) // self.batch_size

# TODO: get_data function to avoid direct initialisation
# TODO: K-fold cross validation using dataloader and custom indexes


