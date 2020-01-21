import os
import re
import json
import numpy as np
import spacy
import gensim
import argparse
import itertools
import multiprocessing as mp
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
parser.add_argument(
    "--dataset-path",
    type=str,
    help="path to dataset",
    default="./dataset/computers/train/computers_splitted_train_xlarge.json",
)
parser.add_argument(
    "--preprocess-method", type=str, help="nltk or spacy preprocess", default="spacy",
)
parser.add_argument(
    "--embed-algorithm",
    type=int,
    help="training algorithm: CBOW (0) or SKIPGRAM (1)",
    default=1,
)
args = parser.parse_args()
preprocess_method = args.preprocess_method

attrs = ["title"]
attributes = [attr + "_left" for attr in attrs] + [attr + "_right" for attr in attrs]
print("* LOADING DATASET")
dataset = np.concatenate(
    [
        parse_content_line(x, attributes=attributes, label=0)
        for x in open(args.dataset_path, "r").readlines()
    ],
    axis=0,
).astype(object)
print("* DONE")
sentences = list(itertools.chain(*dataset))

cores = mp.cpu_count() - 4  # Count the number of cores in a computer
print("* PREPROCESS")
# Preprocess text
if preprocess_method == "spacy":
    # Load spacy for tokenizing text
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    txt = [
        preprocess(doc, method=preprocess_method)
        for doc in nlp.pipe(sentences, batch_size=5000, n_threads=cores)
    ]
elif preprocess_method == "nltk":
    with mp.Pool(processes=cores) as pool:
        for attr in range(len(attributes)):
            txt = pool.map(preprocess, sentences)
print("* DONE")

# Remove duplicates
cleaned_sentences = pd.DataFrame({"sentences": txt})
cleaned_sentences = cleaned_sentences.dropna().drop_duplicates()

# Prepare sentences for w2v/fasttext training
sentences = [row.split() for row in cleaned_sentences["sentences"]]

# Train W2V or FastText
print("* TRAIN EMBEDDINGS")
size = 150
min_count = 1
context_window = 9
epochs = 300
algorithm = "w2v"
if algorithm == "w2v":
    model = Word2Vec(
        sg=args.embed_algorithm,  #  Use SKIPGRAM model
        hs=0,  # Don't use hierarchical softmax
        min_count=min_count,  # All words that have an absolute frequency < 20 will be discarded
        window=context_window,  # Context-window size
        size=size,  # Embeddings dimension
        sample=1e-5,
        alpha=0.03,
        min_alpha=0.0007,
        negative=10,  # How many negative samples will be sampled for each positive example
        workers=cores - 1,
        compute_loss=True,
    )
elif algorithm == "fasttext":
    model = FastText(
        sg=args.embed_algorithm,  #  Use SKIPGRAM model
        hs=0,  # Don't use hierarchical softmax
        min_count=min_count,  # All words that have an absolute frequency < 20 will be discarded
        window=context_window,  # Context-window size
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
model.train(sentences, total_examples=model.corpus_count, epochs=epochs, report_delay=1)
print("* DONE")

if algorithm == "w2v":
    sentences_dict = {}
    with open(
        "./dataset/embeddings/w2v/w2v_"
        + "_".join(attrs)
        + "_"
        + str(epochs)
        + "Epochs_"
        + str(context_window)
        + "ContextWindow_"
        + str(min_count)
        + "MinCount_"
        + str(size)
        + "d.txt",
        "w",
    ) as f:
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
        "./dataset/embeddings/w2v/w2v_"
        + "_".join(attrs)
        + "_"
        + str(epochs)
        + "Epochs_"
        + str(context_window)
        + "ContextWindow_"
        + str(min_count)
        + "MinCount_"
        + str(size)
        + "d.bin",
        binary=True,
    )
elif algorithm == "fasttext":
    sentences_dict = {}
    with open(
        "./dataset/embeddings/fasttext/fasttext_"
        + "_".join(attrs)
        + "_"
        + str(epochs)
        + "Epochs_"
        + str(context_window)
        + "ContextWindow_"
        + str(min_count)
        + "MinCount_"
        + str(size)
        + "d.txt",
        "w",
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
    model.wv.save(
        "./dataset/embeddings/fasttext/fasttext_"
        + "_".join(attrs)
        + "_"
        + str(epochs)
        + "Epochs_"
        + str(context_window)
        + "ContextWindow_"
        + str(min_count)
        + "MinCount_"
        + str(size)
        + "d.bin"
    )
