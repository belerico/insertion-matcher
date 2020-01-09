from dataset import get_data
from models import get_deep_cross_model
import argparse
from utils import dot_similarity, get_pretrained_embedding

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--dataset-path', type=str, help='path to dataset')
parser.add_argument('--pretrained-embeddings-path', type=str, help='path to pretrained embedding')

args = parser.parse_args()

DATA_PATH = args.dataset_path
PRETRAINED_EMBEDDING_PATH = args.pretrained_embeddings_path
NUM_WORDS = 1000
MAX_LEN = 20
BATCH_SIZE = 32
EMBEDDING_DIM = 100

if __name__ == '__main__':
    print('Loading data')
    train_gen, val_gen, word_index = get_data(DATA_PATH, NUM_WORDS, MAX_LEN, BATCH_SIZE,
                                              train_test_split=0.8)

    matrix_similarity_function = dot_similarity
    embedding_matrix = None

    if PRETRAINED_EMBEDDING_PATH != "":
        embedding_matrix = get_pretrained_embedding(PRETRAINED_EMBEDDING_PATH, NUM_WORDS + 1,
                                                    EMBEDDING_DIM, word_index)

    # TODO: create function 'fitting' (useful for HPO)
    model = get_deep_cross_model(NUM_WORDS + 1, EMBEDDING_DIM, MAX_LEN, matrix_similarity_function,
                                 embedding_matrix)
    model.summary()
    model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Start training")
    model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=2)
