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


t

# TODO: implement all similarity functions from paper
