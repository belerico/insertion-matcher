import spacy
import json
import numpy as np
import itertools
import multiprocessing as mp

from pandas import pandas
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences

from utils import preprocess
from utils import parse_content_line


dataset_path = [
    "dataset/computers/train/computers_train_xlarge.json",
    "dataset/computers/test/computers_gs.json",
]
cores = mp.cpu_count()
preprocess_method = "nltk"

attrs = ["title"]
attributes = [attr + "_left" for attr in attrs] + [attr + "_right" for attr in attrs]
print("* LOADING DATASET")
dataset = []

for path in dataset_path:
    with open(path, "r") as f:
        for line in f:
            dataset.append(parse_content_line(line, attributes, label=False))

dataset = np.concatenate(dataset, axis=0).astype(object)
print("* DONE")
sentences = list(itertools.chain(*dataset))

cores = mp.cpu_count()  # Count the number of cores in a computer
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
cleaned_sentences = pandas.DataFrame({"sentences": txt})
cleaned_sentences = cleaned_sentences.dropna().drop_duplicates()

# Prepare sentences for w2v/fasttext training
sentences = list(
    itertools.chain(*[row.split() for row in cleaned_sentences["sentences"]])
)

# List of all unique words, sorted by frequency
word_index = {
    word: (idx + 1) for idx, word in enumerate(list(Counter(sentences).keys()))
}
print("* FOUND", len(word_index), "unique vocabs")

with open("dataset/" + "_".join(attrs) + "_word_index.json", "w") as f:
    f.write(json.dumps(word_index))
