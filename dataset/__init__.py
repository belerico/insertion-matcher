import json
import string

import numpy as np
import torch
from nltk import word_tokenize
from torchtext.data import Field, Dataset, Example, BucketIterator
import numpy as np
from pandas import pandas
from nltk.corpus import stopwords
import re


def preprocess(doc, method="nltk", dataset=True):
    english_stopwords = set(stopwords.words("english"))
    non_alphanum_regex = re.compile(r"\W+")
    if method == "spacy":
        tokens = " ".join(
            [
                token.lower_
                for token in doc
                if token
                and not (token.lower_ == "null" or token.is_stop or token.is_punct)
            ]
        )
    elif method == "nltk":
        # doc = non_alphanum_regex.sub(' ', doc).lower()
        tokens = [
            token
            for token in word_tokenize(doc.lower())
            if not (
                token == "null"
                or token in english_stopwords
                or token in string.punctuation
            )
        ]
    if dataset or tokens != "":
        return tokens


def parse_content_line(x, attributes=None, label=True):
    if attributes is None:
        attributes = ["title_left", "title_right"]
    item = json.loads(x)
    elements = [item[attr] if item[attr] is not None else "" for attr in attributes]
    if label:
        elements.append(int(item["label"]))
    item = np.array(elements)
    return item[np.newaxis, :]


class DataFrameDataset(Dataset):
    """Class for using pandas DataFrames as a datasource"""

    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        self.examples = examples.apply(
            SeriesExample.fromSeries, args=(fields,), axis=1
        ).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]


class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, field in fields.items():
            if key not in data:
                raise ValueError(
                    "Specified key {} was not found in " "the input data".format(key)
                )
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex


class BatchWrapper:
    def __init__(self, dl, x_vars, y_var):
        self.dl, self.x_vars, self.y_var = (
            dl,
            x_vars,
            y_var,
        )  # we pass in the list of attributes for x

    def __iter__(self):
        for batch in self.dl:
            left = getattr(
                batch, self.x_vars[0]
            )  # we assume only one input in this wrapper
            right = getattr(
                batch, self.x_vars[1]
            )  # we assume only one input in this wrapper
            y = torch.Tensor(list(map(float, getattr(batch, self.y_var))))

            yield (left, right, y)

    def __len__(self):
        return len(self.dl)


def get_data(train_path, valid_path, test_path):
    TEXT = Field(
        sequential=True,
        tokenize=preprocess,
        lower=True,
        fix_length=20,
        batch_first=True,
    )
    LABEL = Field(sequential=False, use_vocab=False, is_target=True, batch_first=True)

    contents = []
    for i, x in enumerate(open(train_path, "r").readlines()):
        try:
            item = parse_content_line(x, attributes=None, label=True)
            contents.append(item)
        except:
            print("Lost data at line {}".format(i))

    contents = np.concatenate(contents, axis=0)
    train = pandas.DataFrame(
        data=contents, columns=["title_left", "title_right", "label"]
    )

    contents = []
    for i, x in enumerate(open(valid_path, "r").readlines()):
        try:
            item = parse_content_line(x, attributes=None, label=True)
            contents.append(item)
        except:
            print("Lost data at line {}".format(i))

    contents = np.concatenate(contents, axis=0)
    valid = pandas.DataFrame(
        data=contents, columns=["title_left", "title_right", "label"]
    )

    contents = []
    for i, x in enumerate(open(test_path, "r").readlines()):
        try:
            item = parse_content_line(x, attributes=None, label=True)
            contents.append(item)
        except:
            print("Lost data at line {}".format(i))

    contents = np.concatenate(contents, axis=0)
    test = pandas.DataFrame(
        data=contents, columns=["title_left", "title_right", "label"]
    )

    fields = {"title_left": TEXT, "title_right": TEXT, "label": LABEL}
    train_ds = DataFrameDataset(train, fields)
    valid_ds = DataFrameDataset(valid, fields)
    test_ds = DataFrameDataset(test, fields)
    TEXT.build_vocab(train_ds, valid_ds)
    print(train_ds[0].title_left)
    print(valid_ds[0].title_left)
    print(test_ds[0].title_left)

    return (
        train_ds,
        valid_ds,
        test_ds,
        TEXT,
    )


def get_iterators(train_ds, valid_ds, test_ds):
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train_ds, valid_ds, test_ds),
        # we pass in the datasets we want the iterator to draw data from
        batch_sizes=(32, 64, 64),
        device=torch.device(
            "cpu"
        ),  # if you want to use the GPU, specify the GPU number here
        sort_key=lambda x: min(max(len(x.title_left), len(x.title_right)), 20),
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=True,
        shuffle=True,
        repeat=False,  # we pass repeat=False because we want to wrap this Iterator layer.
    )

    train_dl = BatchWrapper(train_iter, ["title_left", "title_right"], "label")
    valid_dl = BatchWrapper(val_iter, ["title_left", "title_right"], "label")
    test_dl = BatchWrapper(test_iter, ["title_left", "title_right"], "label")
    return train_dl, valid_dl, test_dl
