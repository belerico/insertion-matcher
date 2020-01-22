import utils
import functools
import operator
from fitness import fit
from dataset import get_data, get_wdc_data
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import json
from utils import get_pretrained_embedding
# # Execution


TRAIN_PATH = './dataset/computers/train/computers_train_large.json'
VALID_PATH = './dataset/computers/valid/computers_splitted_valid_medium.json'
TEST_PATH = './dataset/computers/test/computers_gs.json'
PRETRAINED_EMBEDDING_PATH = \
    './dataset/embeddings/fasttext/fasttext_title_300Epochs_9ContextWindow_1MinCount_100d.txt'
NUM_WORDS = None
MAX_LEN = 20
BATCH_SIZE = 64
EMBEDDING_DIM = 100
EPOCHS = 100
EXP_DIR = './experiments/large/fasttext_200'
WDC = True

import pickle

with open('./data/exp' + str(EMBEDDING_DIM) + 'embedding_w2v.pickle', 'rb') as f:
    params = pickle.load(f)
max_f1 = 0
best_params = {}
for param in params:
    if param[1] > max_f1:
        best_params = param[0]
        max_f1 = param[1]
print(best_params)

LR = best_params['lr']
RNN_UNITS = best_params['rnn_units']
SIMILARITY_FUNC = 'dot_similarity' if best_params['similarity_type'] == 0 else 'cosine_similarity'
CONVS_FILTER_BANKS = best_params['convs_filter_banks']
CONVS_KERNEL_SIZE = best_params['convs_kernel_size']
POOL_SIZE = best_params['pool_size']
DENSES_DEPTH = best_params['denses_depth']

if __name__ == "__main__":
    print("* LOADING DATA")
    if not WDC:
        train_gen, val_gen, word_index, class_weights = get_data(
            TRAIN_PATH,
            word_index_path=None,
            num_words=NUM_WORDS,
            max_len=MAX_LEN,
            batch_size=BATCH_SIZE,
            preprocess_data=True,
            preprocess_method="nltk",
            train_test_split=0.8,
        )
    else:
        train_gen, val_gen, test_gen, class_weights = get_wdc_data(
            TRAIN_PATH,
            VALID_PATH,
            TEST_PATH,
            word_index_path="dataset/title_word_index.json",
            num_words=NUM_WORDS,
            max_len=MAX_LEN,
            batch_size=BATCH_SIZE,
            preprocess_data=True,
            preprocess_method="nltk",
        )
        word_index = json.loads(open("dataset/title_word_index.json").read())

    NUM_WORDS = len(word_index) if NUM_WORDS is None else NUM_WORDS
    print("* NUM WORDS: ", NUM_WORDS)
    print("* CLASS WEIGHTS:", class_weights)

    embedding_matrix = None
    if PRETRAINED_EMBEDDING_PATH is not None:
        embedding_matrix = get_pretrained_embedding(
            PRETRAINED_EMBEDDING_PATH, NUM_WORDS + 1, EMBEDDING_DIM, word_index
        )
    matrix_similarity_function = getattr(utils, SIMILARITY_FUNC)
    model, results = fit(
        train_gen,
        val_gen,
        NUM_WORDS,
        EMBEDDING_DIM,
        MAX_LEN,
        matrix_similarity_function,
        EXP_DIR,
        10,
        denses_depth=DENSES_DEPTH,
        activation="sigmoid",
        embedding_matrix=embedding_matrix,
        embedding_trainable=False,
        embedding_dropout=0.3,
        rnn_type="LSTM",
        rnn_units=RNN_UNITS,
        convs_filter_banks=CONVS_FILTER_BANKS,
        convs_kernel_size=CONVS_KERNEL_SIZE,
        pool_size=POOL_SIZE,
        mlp_dropout=0.3,
        epochs=EPOCHS,
        verbosity=2,
        callbacks=True,
        class_weights=None,
        optimizer=None
    )

y_true = [v[1] for v in test_gen]
y_true = functools.reduce(operator.iconcat, y_true, [])
predictions = model.predict(test_gen) > 0.5
print(classification_report(y_true, predictions))
