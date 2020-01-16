from dataset import get_data
import argparse
from utils import dot_similarity, get_pretrained_embedding, cosine_similarity
from fitness import fit

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument(
    "--dataset-path",
    type=str,
    help="path to dataset",
    default="./dataset/computers/train/computers_splitted_train_medium.json",
)
parser.add_argument("--exp-path", type=str, help="path to exp", default="./experiments")
parser.add_argument(
    "--pretrained-embeddings-path",
    type=str,
    help="path to pretrained embedding",
    default="./dataset/embeddings/w2v/w2v_title_brand_description_1MinCount_5ContextWindow_100d.txt",
)

args = parser.parse_args()

DATA_PATH = args.dataset_path
EXP_DIR = args.exp_path
PRETRAINED_EMBEDDING_PATH = args.pretrained_embeddings_path
NUM_WORDS = None
MAX_LEN = 20
BATCH_SIZE = 32
EMBEDDING_DIM = 100
EARLY_STOPPING = 10
CONVS_DEPTH = [16]
DENSES_DEPTH = [32, 16]

if __name__ == "__main__":
    print("* LOADING DATA")
    train_gen, val_gen, word_index = get_data(
        DATA_PATH,
        NUM_WORDS,
        MAX_LEN,
        BATCH_SIZE,
        train_test_split=0.8,
        preprocess_method="nltk",
    )
    NUM_WORDS = len(word_index) if NUM_WORDS is None else NUM_WORDS
    print("* NUM WORDS: ", NUM_WORDS)
    matrix_similarity_function = cosine_similarity
    embedding_matrix = None

    if PRETRAINED_EMBEDDING_PATH is not None:
        embedding_matrix = get_pretrained_embedding(
            PRETRAINED_EMBEDDING_PATH, NUM_WORDS + 1, EMBEDDING_DIM, word_index
        )

    model = fit(
        train_gen,
        val_gen,
        NUM_WORDS,
        EMBEDDING_DIM,
        MAX_LEN,
        matrix_similarity_function,
        EXP_DIR,
        EARLY_STOPPING,
        embedding_matrix=embedding_matrix,
        embedding_trainable=True,
        convs_depth=CONVS_DEPTH,
        denses_depth=DENSES_DEPTH,
        dropout=True,
        activation="tanh",
    )
