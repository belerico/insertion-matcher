import os
import re
import json
import spacy
import gensim
import argparse
import itertools
import multiprocessing
from gensim.models import Word2Vec
from pandas import pandas as pd

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description='Train Word2Vec model')
parser.add_argument('--dataset-path', type=str, help='path to dataset')
parser.add_argument('--algorithm', type=str, help='training algorithm: CBOW or SKIPGRAM')
parser.add_argument('--pretrained-embeddings-path', type=str, help='path to pretrained embedding')
args = parser.parse_args()

# Load spacy for tokenizing text
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

def preprocess(doc): 
    tokens = ' '.join([token.lower_.strip() for token in doc if token and not (token.lower_.strip() == 'null' or token.is_stop or token.is_punct)])
    if tokens != '':
        return tokens
    
dataset = pd.read_json('./dataset/all_train_small.json')
attributes = ['title', 'brand', 'description']
attributes = [attr + '_left' for attr in attributes] + [attr + '_right' for attr in attributes]
# Replace None value with empty string
dataset[attributes] = dataset[attributes].fillna('')
sentences = list(itertools.chain(*dataset[attributes].values.tolist()))
# Preprocess text
txt = [preprocess(doc) for doc in nlp.pipe(sentences, batch_size=5000, n_threads=4)]
# Remove duplicates
cleaned_sentences = pd.DataFrame({'sentences': txt})
cleaned_sentences = cleaned_sentences.dropna().drop_duplicates()
# Prepare sentences for w2v training
sentences = [row.split() for row in cleaned_sentences['sentences']]
# Train W2V
size = 150
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model = Word2Vec(
    sg=1, #  Use SKIPGRAM model
    hs=0,  # Don't use hierarchical softmax
    min_count=20,  # All words that have an absolute frequency < 20 will be discarded
    window=3,  # Context-window size
    size=size,  # Embeddings dimension
    sample=1e-5, 
    alpha=0.03, 
    min_alpha=0.0007, 
    negative=10,  # How many negative samples will be sampled for each positive example
    workers=cores-1
)
w2v_model.build_vocab(sentences, progress_per=10000)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
w2v_model.wv.save_word2vec_format('./dataset/w2v_' + str(size) + '.bin', binary=True)