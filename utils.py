import json

import numpy as np
from keras import backend as K
from keras.backend.tensorflow_backend import tf_math_ops


def parse_content_line(x):
    item = json.loads(x)
    item = np.array([item['title_left'], item['title_right'], int(item['label'])])
    return item[np.newaxis, :]


def dot_similarity(tensor1, tensor2):
    matrix = tf_math_ops.batch_mat_mul(tensor1, tensor2, adj_y=True)
    matrix = K.expand_dims(matrix, axis=-1)
    return matrix


def get_pretrained_embedding(embedding_file, num_words, embedding_dimension, word_index):
    embeddings_index = {}
    with open(embedding_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((num_words, embedding_dimension))

    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# TODO: implement all similarity functions from paper
