import json
import spacy
import numpy as np
import itertools
import multiprocessing as mp

from pandas import pandas
from collections import Counter
from sklearn.utils import class_weight
from keras.preprocessing.text import Tokenizer
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences

from utils import preprocess
from utils import parse_content_line


# TODO: to work with variable length sentences, we need to pad, within the batch, all the
# sentences that are shorter than the maximum-length sentence
# model = Sequential()
# model.add(Masking(mask_value=0., input_shape=(None, 10)))
# model.add(LSTM(32))


class Dataset:
    def __init__(
            self,
            data_path,
            word_index_path=None,
            attributes=None,
            preprocess_data=True,
            preprocess_method="nltk",  # or spacy
            num_words=None,
            max_len=20,
    ):
        if attributes is None:
            attributes = ["title_left", "title_right"]

        contents = []
        for i, x in enumerate(open(data_path, "r").readlines()):
            try:
                item = parse_content_line(x, attributes=attributes, label=True)
                contents.append(item)
            except:
                print("Lost data at line {}".format(i))

        contents = np.concatenate(contents, axis=0)
        if preprocess_data:
            print("* PREPROCESS DATA")
            if preprocess_method == "spacy":
                nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
                for attr in range(len(attributes)):
                    contents[:, attr] = [
                        preprocess(doc, method="spacy")
                        for doc in nlp.pipe(
                            contents[:, attr].astype(object),
                            batch_size=5000,
                            n_threads=4,
                        )
                    ]
                # del contents_df
            elif preprocess_method == "nltk":
                with mp.Pool(processes=4) as pool:
                    for attr in range(len(attributes)):
                        contents[:, attr] = pool.map(preprocess, contents[:, attr])
            print("* DONE")

        if word_index_path is None:
            cleaned_sentences = list(itertools.chain(*contents[:, : len(attributes)]))
            # Create a word index on our own
            cleaned_sentences = list(
                itertools.chain(*list(map(str.split, cleaned_sentences)))
            )
            # List of all unique words, sorted by frequency
            self.word_index = {
                word: (idx + 1)
                for idx, word in enumerate(list(Counter(cleaned_sentences).keys()))
            }
            print("* FOUND", len(self.word_index), "unique vocabs")
            del cleaned_sentences
        else:
            self.word_index = json.load(open(word_index_path).read())

        self.dataset = np.zeros([len(contents), 2, max_len])
        self.labels = contents[:, len(attributes)].astype(int)
        print(self.labels)

        # Tokenize sentences (words -> indices)
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.word_index = self.word_index
        self.dataset[:, 0, :] = pad_sequences(
            tokenizer.texts_to_sequences(contents[:, 0]),
            max_len,
            padding="post",
            truncating="post",
        )
        self.dataset[:, 1, :] = pad_sequences(
            tokenizer.texts_to_sequences(contents[:, 1]),
            max_len,
            padding="post",
            truncating="post",
        )
        del tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        item = self.dataset[key]
        first = item[0]
        second = item[1]
        label = np.array([int(self.labels[key])])
        return first, second, label


class Dataloader(Sequence):
    def __init__(
            self, dataset, batch_size=32, indexes: np.ndarray = None, shuffle=False
    ):
        self.dataset = dataset
        self.indexes = indexes if indexes is not None else np.arange(len(self.dataset))

        np.random.shuffle(self.indexes)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __getitem__(self, key):
        ids = key * self.batch_size
        idf = ids + self.batch_size if ids < len(self.dataset) else self.dataset
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
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.indexes) // self.batch_size


def get_train_test_indexes(max_index, train_test_split):
    indexes = np.arange(max_index)
    np.random.shuffle(indexes)
    train_split = int(max_index * train_test_split)
    return indexes[:train_split], indexes[train_split:]


def get_wdc_data(
        train_path,
        valid_path,
        test_path,
        word_index_path=None,
        num_words=None,
        max_len=20,
        batch_size=32,
        preprocess_data=True,
        preprocess_method="nltk",
):
    train = Dataset(
        train_path,
        num_words=num_words,
        max_len=max_len,
        preprocess_data=preprocess_data,
        preprocess_method=preprocess_method,
    )
    train_gen = Dataloader(train, batch_size, None, shuffle=True)
    valid = Dataset(
        valid_path,
        num_words=num_words,
        max_len=max_len,
        preprocess_data=preprocess_data,
        preprocess_method=preprocess_method,
    )
    valid_gen = Dataloader(
        valid,
        batch_size,
        None,
        shuffle=True
    )
    test_gen = Dataloader(
        Dataset(
            test_path,
            num_words=num_words,
            max_len=max_len,
            preprocess_data=preprocess_data,
            preprocess_method=preprocess_method,
        ),
        batch_size,
        None,
    )
    class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(train.labels), train.labels
    )
    return train_gen, valid_gen, test_gen, class_weights


def get_data(
        data_path,
        word_index_path=None,
        num_words=None,
        max_len=20,
        batch_size=32,
        preprocess_data=True,
        preprocess_method="nltk",
        train_test_split=0.8,
):
    dataset = Dataset(
        data_path,
        num_words=num_words,
        max_len=max_len,
        preprocess_data=preprocess_data,
        preprocess_method=preprocess_method,
    )
    train_indexes, test_indexes = get_train_test_indexes(len(dataset), train_test_split)
    train_gen = Dataloader(dataset, batch_size, train_indexes)
    val_gen = Dataloader(dataset, batch_size, test_indexes)
    class_weights = class_weight.compute_class_weight(
        "balanced", np.unique(train_gen.dataset.labels), train_gen.dataset.labels
    )
    return train_gen, val_gen, dataset.word_index, class_weights


def get_kfold_generator(data_path, num_words, max_len, batch_size, n_folds):
    dataset = Dataset(data_path, num_words=num_words, max_len=max_len)
    n_samples = len(dataset)
    indexes = np.arange(n_samples)
    np.random.shuffle(indexes)
    for i in range(n_folds):
        fold_dim = int(n_samples / n_folds)
        test_indexes = indexes[fold_dim * i: fold_dim * (i + 1)]
        train_indexes = np.setdiff1d(indexes, test_indexes)
        train_gen = Dataloader(dataset, batch_size, train_indexes)
        val_gen = Dataloader(dataset, batch_size, test_indexes)
        yield train_gen, val_gen
