import json
import spacy
import numpy as np
from keras import backend as K
from keras.backend.tensorflow_backend import tf_math_ops


def preprocess(doc: spacy.tokens.Doc, dataset=True):
    tokens = " ".join(
        [
            token.lower_.strip()
            for token in doc
            if token
            and not (token.lower_.strip() == "null" or token.is_stop or token.is_punct)
        ]
    )
    if dataset or tokens != '':
        return tokens


def parse_content_line(x, attributes=['title_left', 'title_right'], label=True):
    item = json.loads(x)
    elements = [item[attr] for attr in attributes]
    if label:
        elements.append(int(item['label']))
    item = np.array(elements)
    return item[np.newaxis, :]


def dot_similarity(tensor1, tensor2):
    matrix = tf_math_ops.batch_mat_mul(tensor1, tensor2, adj_y=True)
    matrix = K.expand_dims(matrix, axis=-1)

    return matrix

def get_pretrained_embedding(embedding_file, num_words, embedding_dimension, word_index, model='w2v'):
    # here we load the pretraiend embedding
    embeddings_index = {}
    with open(embedding_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1) # load embedding representations
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    # final matrix embedding with Dataset indexes
    embedding_matrix = np.zeros((num_words if model == 'glove' else len(embeddings_index), embedding_dimension))

    # word index of Dataset instance (mapping of vocabs and their indexes)
    for word, i in word_index.items():

        if i >= num_words: # if index i is out of bound, skip it
            continue

        # get word representation from previous loaded embedding
        embedding_vector = embeddings_index.get(word)

        # if the word has a representation in embeddings, create the association (i -> embedding)
        # in final embedding_matrix
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


# TODO: implement all similarity functions from paper
