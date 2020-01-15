import os
import re
import json
import numpy as np
import spacy
import gensim
import argparse
import itertools
import multiprocessing
from gensim.models import Word2Vec, FastText
from pandas import pandas as pd

from utils import preprocess
from utils import parse_content_line

import logging  # Setting up the loggings to monitor gensim

logging.basicConfig(
    format="%(levelname)s - %(asctime)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

parser = argparse.ArgumentParser(description="Train Word2Vec model")
parser.add_argument("--dataset-path", type=str, help="path to dataset")
parser.add_argument(
    "--algorithm", type=str, help="training algorithm: CBOW or SKIPGRAM"
)
args = parser.parse_args()

# Load spacy for tokenizing text
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

attrs = ["title", "brand", "description"]
attributes = [attr + "_left" for attr in attrs] + [attr + "_right" for attr in attrs]
dataset = np.concatenate(
    [
        parse_content_line(x, attributes=attributes, label=0)
        for x in open("./dataset/all_train_small.json", "r").readlines()
    ],
    axis=0,
)
dataset = pd.DataFrame(data=dataset, columns=attributes)

# Replace None value with empty string
dataset[attributes] = dataset[attributes].fillna("")
sentences = list(itertools.chain(*dataset[attributes].values.tolist()))

# Preprocess text
txt = [preprocess(doc) for doc in nlp.pipe(sentences, batch_size=5000, n_threads=4)]

# Remove duplicates
cleaned_sentences = pd.DataFrame({"sentences": txt})
cleaned_sentences = cleaned_sentences.dropna().drop_duplicates()

# Prepare sentences for w2v/fasttext training
sentences = [row.split() for row in cleaned_sentences["sentences"]]

# Train W2V or FastText
# Train W2V or FastText
size = 150
cores = multiprocessing.cpu_count()  # Count the number of cores in a computer
algorithm = "fasttext"
if algorithm == "w2v":
    model = Word2Vec(
        sg=1,  #  Use SKIPGRAM model
        hs=0,  # Don't use hierarchical softmax
        min_count=9,  # All words that have an absolute frequency < 20 will be discarded
        window=7,  # Context-window size
        size=size,  # Embeddings dimension
        sample=1e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=10,  # How many negative samples will be sampled for each positive example
        workers=cores - 1,
    )
elif algorithm == "fasttext":
    model = FastText(
        sg=1,  #  Use SKIPGRAM model
        hs=0,  # Don't use hierarchical softmax
        min_count=5,  # All words that have an absolute frequency < 20 will be discarded
        window=9,  # Context-window size
        size=size,  # Embeddings dimension
        sample=1e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=10,  # How many negative samples will be sampled for each positive example
        workers=cores - 1,
        word_ngrams=1,
        min_n=3,
        max_n=6,
    )

model.build_vocab(sentences, progress_per=10000)
model.train(sentences, total_examples=model.corpus_count, epochs=30, report_delay=1)

if algorithm == "w2v":
    sentences_dict = {}
    with open("./dataset/w2v_" + "_".join(attrs) + "_" + str(size) + ".txt", "w") as f:
        for sentence in sentences:
            for token in sentence:
                try:
                    sentences_dict[token]
                except KeyError:
                    sentences_dict[token] = 1
                    if token in model.wv.vocab:
                        f.write(
                            token
                            + " "
                            + " ".join(str(x) for x in model.wv.get_vector(token))
                        )
                        f.write("\n")
        f.flush()
        f.close()
    model.wv.save_word2vec_format(
        "./dataset/w2v_" + "_".join(attrs) + "_" + str(size) + ".bin", binary=True
    )
elif algorithm == "fasttext":
    sentences_dict = {}
    with open(
        "./dataset/fasttext_" + "_".join(attrs) + "_" + str(size) + ".txt", "w"
    ) as f:
        for sentence in sentences:
            for token in sentence:
                try:
                    sentences_dict[token]
                except KeyError:
                    sentences_dict[token] = 1
                    try:
                        f.write(
                            token
                            + " "
                            + " ".join(str(x) for x in model.wv.get_vector(token))
                        )
                        f.write("\n")
                    except KeyError:
                        continue
        f.flush()
        f.close()
    model.wv.save("./dataset/fasttext_" + "_".join(attrs) + "_" + str(size) + ".bin")
