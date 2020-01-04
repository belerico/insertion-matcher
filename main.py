from dataset import Dataset, Dataloader
import keras
from models import get_deep_cross_model
import argparse
from utils import dot_similarity

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--dataset-path', type=str, help='path to dataset')

args = parser.parse_args()

DATA_PATH = args.dataset_path
DATA_PATH = 'dataset/all_train_small.json'
NUM_WORDS = 1000
MAX_LEN = 20
BATCH_SIZE = 32
EMBEDDING_DIM = 10

if __name__ == '__main__':
    print('Loading data')
    dataset = Dataset(DATA_PATH, num_words=NUM_WORDS, max_len=MAX_LEN)
    dataloader = Dataloader(dataset, BATCH_SIZE)
    matrix_similarity_function = dot_similarity

    # TODO: create function 'fitting' (useful for HPO)
    model = get_deep_cross_model(NUM_WORDS, EMBEDDING_DIM, MAX_LEN, matrix_similarity_function)
    model.summary()
    model.compile('adam', loss='binary_crossentropy')
    model.fit(dataloader)
