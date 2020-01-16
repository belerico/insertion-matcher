import re
import json
import spacy
import numpy as np
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from keras import backend as K
from keras.layers import dot
from keras.backend.tensorflow_backend import tf_math_ops

english_stopwords = set(stopwords.words("english"))
non_alphanum_regex = re.compile(r'\W+')

def preprocess(doc, method='nltk', dataset=True):
    if method == 'spacy':
        tokens = " ".join(
            [
                token.lower_
                for token in doc
                if token
                and not (token.lower_ == "null" or token.is_stop or token.is_punct)
            ]
        )
    elif method == 'nltk':
        # doc = non_alphanum_regex.sub(' ', doc).lower()
        tokens = " ".join(
            [
                token
                for token in word_tokenize(doc.lower())
                if not (token == "null" or token in english_stopwords or token in string.punctuation)
            ]
        )
    if dataset or tokens != "":
        return tokens


def parse_content_line(x, attributes=None, label=True):
    if attributes is None:
        attributes = ["title_left", "title_right"]
    item = json.loads(x)
    elements = [item[attr] if item[attr] is not None else '' for attr in attributes]
    if label:
        elements.append(int(item["label"]))
    item = np.array(elements)
    return item[np.newaxis, :]


def dot_similarity(tensor1, tensor2):
    matrix = tf_math_ops.batch_mat_mul(tensor1, tensor2, adj_y=True)
    matrix = K.expand_dims(matrix, axis=-1)
    return matrix


def cosine_similarity(tensor1, tensor2):
    matrix = tf_math_ops.batch_mat_mul(
        K.l2_normalize(tensor1, axis=-1), K.l2_normalize(tensor2, axis=-1), adj_y=True
    )
    matrix = K.expand_dims(matrix, axis=-1)
    return matrix


def get_pretrained_embedding(
        embedding_file, num_words, embedding_dimension, word_index
):
    print("* LOADING EMBEDDINGS MATRIX")
    # here we load the pretraiend embedding
    embeddings_index = {}
    with open(embedding_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)  # load embedding representations
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("* FOUND %s WORD VECTORS" % len(embeddings_index))

    # final matrix embedding with Dataset indexes
    embedding_matrix = np.zeros((num_words, embedding_dimension))

    words_found = 0
    # word index of Dataset instance (mapping of vocabs and their indexes)
    for word, i in word_index.items():

        # get word representation from previous loaded embedding
        embedding_vector = embeddings_index.get(word)

        # if the word has a representation in embeddings, create the association (i -> embedding)
        # in final embedding_matrix
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            words_found += 1
    print("* FOUND", words_found, "vector representations out of", num_words, "words")

    return embedding_matrix

# TODO: implement all similarity functions from paper
