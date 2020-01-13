from keras.models import Model
from keras.layers import Input, Embedding, Conv2D, Flatten, Dense, Bidirectional, LSTM, Lambda, \
    MaxPool2D
from keras.initializers import Constant


def gen_interaction_matrix(matrix_similarity_function):
    def fun(inputs_list):
        bi_left = inputs_list[0]
        bi_right = inputs_list[1]
        matrix = matrix_similarity_function(bi_left, bi_right)

        return matrix

    return Lambda(fun)


def get_deep_cross_model(vocab_size, embedding_dimension, vec_dimension,
                         matrix_similarity_function, convs_depth, denses_depth, activation,
                         trainable=False,
                         embedding_matrix=None):
    left_input = Input((vec_dimension,))
    right_input = Input((vec_dimension,))

    if embedding_matrix is None:
        embed = Embedding(vocab_size, embedding_dimension)
    else:
        embed = Embedding(vocab_size, embedding_dimension,
                          embeddings_initializer=Constant(embedding_matrix),
                          trainable=trainable)

    left_encoded = embed(left_input)
    right_encoded = embed(right_input)

    bi_left = Bidirectional(LSTM(embedding_dimension, return_sequences=True), merge_mode='concat')(
        left_encoded)

    bi_right = Bidirectional(LSTM(embedding_dimension, return_sequences=True), merge_mode='concat')(
        right_encoded)

    x = gen_interaction_matrix(matrix_similarity_function)([bi_left, bi_right])

    for conv_depth in convs_depth:
        x = Conv2D(conv_depth, (3, 3), activation='relu')(x)
        x = MaxPool2D()(x)

    x = Flatten()(x)
    for dense_depth in denses_depth:
        x = Dense(dense_depth, activation='relu')(x)

    output = Dense(1, activation=activation)(x)

    model = Model(inputs=[left_input, right_input], outputs=[output])
    return model
