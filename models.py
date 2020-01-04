from keras.models import Model
from keras.layers import Input, Embedding, Conv2D, Flatten, Dense, Bidirectional, LSTM, Lambda


def gen_interaction_matrix(matrix_similarity_function):
    def fun(inputs_list):
        bi_left = inputs_list[0]
        bi_right = inputs_list[1]
        matrix = matrix_similarity_function(bi_left, bi_right)

        return matrix

    return Lambda(fun)


def get_deep_cross_model(vocab_size, embedding_dimension, vec_dimension, matrix_similarity_function):
    left_input = Input((vec_dimension,))
    right_input = Input((vec_dimension,))

    embed = Embedding(vocab_size + 1, embedding_dimension)
    left_encoded = embed(left_input)
    right_encoded = embed(right_input)

    bi_left = Bidirectional(LSTM(embedding_dimension, return_sequences=True))(left_encoded)

    bi_right = Bidirectional(LSTM(embedding_dimension, return_sequences=True))(right_encoded)

    interaction_matrix = gen_interaction_matrix(matrix_similarity_function)([bi_left, bi_right])

    x = Conv2D(16, (3, 3), activation='relu')(interaction_matrix)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[left_input, right_input], outputs=[output])
    return model
