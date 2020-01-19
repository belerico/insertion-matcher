from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    Conv2D,
    Flatten,
    Dense,
    Bidirectional,
    LSTM,
    GRU,
    Lambda,
    MaxPool2D,
    Dropout,
    BatchNormalization,
    SimpleRNN
)
from keras.initializers import Constant


def gen_interaction_matrix(matrix_similarity_function):
    def fun(inputs_list):
        bi_left = inputs_list[0]
        bi_right = inputs_list[1]
        matrix = matrix_similarity_function(bi_left, bi_right)

        return matrix

    return Lambda(fun)


def get_deep_cross_model(
    vocab_size,
    embedding_dimension,
    vec_dimension,
    matrix_similarity_function,
    convs_depth,
    denses_depth,
    activation,
    rnn_type='LSTM',
    rnn_dimension=100,
    rnn_dropout=0.3,
    embedding_matrix=None,
    embedding_trainable=False,
    embedding_dropout=0.3,
    mlp_dropout=0.3,
):
    left_input = Input((vec_dimension,))
    right_input = Input((vec_dimension,))

    if embedding_matrix is None:
        embed = Embedding(vocab_size, embedding_dimension)
    else:
        embed = Embedding(
            vocab_size,
            embedding_dimension,
            embeddings_initializer=Constant(embedding_matrix),
            trainable=embedding_trainable,
            mask_zero=True
        )

    left_encoded = embed(left_input)
    right_encoded = embed(right_input)
    if embedding_dropout:
        left_encoded = Dropout(embedding_dropout)(left_encoded)
        right_encoded = Dropout(embedding_dropout)(right_encoded)

    rnn_dropout = rnn_dropout if rnn_dropout else 0
    if rnn_type == 'GRU':
        rnn_left = GRU(rnn_dimension, return_sequences=True, recurrent_dropout=rnn_dropout)
        rnn_right = GRU(rnn_dimension, return_sequences=True, recurrent_dropout=rnn_dropout)
    elif rnn_type == 'LSTM':
        rnn_left = LSTM(rnn_dimension, return_sequences=True, recurrent_dropout=rnn_dropout)
        rnn_right = LSTM(rnn_dimension, return_sequences=True, recurrent_dropout=rnn_dropout)

    bi_left = Bidirectional(rnn_left, merge_mode="concat")(left_encoded)
    bi_right = Bidirectional(rnn_right, merge_mode="concat")(right_encoded)

    x = gen_interaction_matrix(matrix_similarity_function)([bi_left, bi_right])

    for conv_depth in convs_depth:
        x = Conv2D(conv_depth, (2, 2), activation="relu")(x)
        x = MaxPool2D()(x)

    x = Flatten()(x)
    for i, dense_depth in enumerate(denses_depth):
        x = Dense(dense_depth, activation="relu")(x)
        if i < len(denses_depth) - 1 and mlp_dropout:
            x = Dropout(mlp_dropout)(x)

    output = Dense(1, activation=activation)(x)

    model = Model(inputs=[left_input, right_input], outputs=[output])
    return model
