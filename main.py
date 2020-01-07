from dataset import get_data
from models import get_deep_cross_model
import argparse
from utils import dot_similarity

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--dataset-path', type=str, help='path to dataset')

args = parser.parse_args()

DATA_PATH = args.dataset_path
NUM_WORDS = 1000
MAX_LEN = 20
BATCH_SIZE = 32
EMBEDDING_DIM = 100

if __name__ == '__main__':
    print('Loading data')
    train_gen, val_gen = get_data(DATA_PATH, NUM_WORDS, MAX_LEN, BATCH_SIZE, train_test_split=0.8)

    matrix_similarity_function = dot_similarity

    # TODO: create function 'fitting' (useful for HPO)
    model = get_deep_cross_model(NUM_WORDS, EMBEDDING_DIM, MAX_LEN, matrix_similarity_function)
    model.summary()
    model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=2)
