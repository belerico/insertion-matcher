import json
import argparse
import functools
import operator
from fitness import fit
from dataset import get_data, get_wdc_data
from sklearn.metrics import classification_report
from utils import dot_similarity, get_pretrained_embedding, cosine_similarity
from keras.backend.tensorflow_backend import softmax

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument(
    "--train-path",
    type=str,
    help="path to train dataset",
    default="./dataset/computers/train/computers_splitted_train_medium.json",
)
parser.add_argument(
    "--valid-path",
    type=str,
    help="path to validation dataset",
    default="./dataset/computers/valid/computers_splitted_valid_medium.json",
)
parser.add_argument(
    "--test-path",
    type=str,
    help="path to test dataset",
    default="./dataset/computers/test/computers_gs.json",
)
parser.add_argument("--exp-path", type=str, help="path to exp", default="./experiments")
parser.add_argument(
    "--pretrained-embeddings-path",
    type=str,
    help="path to pretrained embedding",
    default="./dataset/embeddings/w2v/w2v_title_1MinCount_5ContextWindow_100d.txt",
)

args = parser.parse_args()
DATA_PATH = args.train_path
EXP_DIR = args.exp_path
PRETRAINED_EMBEDDING_PATH = args.pretrained_embeddings_path
NUM_WORDS = None
MAX_LEN = 20
BATCH_SIZE = 32
EMBEDDING_DIM = 100
RNN_UNITS = 200
EARLY_STOPPING = 10
CONVS_DEPTH = 2
DENSES_DEPTH = 2
EPOCHS = 1
wdc = False

if __name__ == "__main__":
    print("* LOADING DATA")
    if not wdc:
        train_gen, val_gen, word_index, class_weights = get_data(
            DATA_PATH,
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
            args.train_path,
            args.valid_path,
            args.test_path,
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
    matrix_similarity_function = cosine_similarity
    model, results = fit(
        train_gen,
        val_gen,
        NUM_WORDS,
        EMBEDDING_DIM,
        MAX_LEN,
        matrix_similarity_function,
        EXP_DIR,
        EARLY_STOPPING,
        convs_depth=1,
        denses_depth=1,
        activation="sigmoid",
        embedding_matrix=None,
        embedding_trainable=False,
        embedding_dropout=0.3,
        rnn_type="LSTM",
        rnn_units=100,
        rnn_dropout=0.3,
        convs_filter_banks=8,
        convs_kernel_size=2,
        pool_size=2,
        denses_units=32,
        mlp_dropout=0.3,
        epochs=EPOCHS,
        verbosity=1,
        callbacks=False,
        class_weights=None,
        optimizer=None
    )
print(results)

y_true = [v[1] for v in val_gen]
y_true = functools.reduce(operator.iconcat, y_true, [])
predictions = model.predict(val_gen) > 0.5
print(predictions)
print(model.predict(val_gen))
print(classification_report(y_true, predictions))
