import os
import re
import json
import gensim
import numpy as np
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from torchtext.vocab import Vectors

english_stopwords = set(stopwords.words("english"))
non_alphanum_regex = re.compile(r"\W+")


def preprocess(doc, method="nltk", dataset=True):
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
        tokens = " ".join(
            [
                token
                for token in word_tokenize(doc.lower())
                if not (
                    token == "null"
                    or token in english_stopwords
                    or token in string.punctuation
                )
            ]
        )
    if dataset or tokens != "":
        return tokens


def parse_content_line(x, attributes=None, label=True):
    if attributes is None:
        attributes = ["title_left", "title_right"]
    item = json.loads(x)
    elements = [item[attr] if item[attr] is not None else "" for attr in attributes]
    if label:
        ll = int(item["label"])
        elements.append(ll)
    item = np.array(elements)
    return item[np.newaxis, :]


def resave_w2v_model(old_path, new_path):
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(old_path, binary=True)
    w2v_model.save_word2vec_format(fname=new_path)


def resave_fasttext_model(old_path, new_path):
    fasttext_model = gensim.models.fasttext.load_facebook_model(old_path)
    fasttext_model.save(fname=new_path)


def load_embedding(TEXT, embedding_path):
    _, file_extension = os.path.splitext(embedding_path)

    if file_extension == ".bin":
        embedding_name = os.path.basename(embedding_path)
        embedding_dir = os.path.dirname(embedding_path)
        vectors = Vectors(name=embedding_name, cache=embedding_dir)
    elif file_extension == ".txt":
        vectors = Vectors(name=embedding_path)
    else:
        raise NotImplementedError()

    TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
