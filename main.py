from dataset import Dataset, Dataloader
import keras

import argparse

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--dataset-path', type=str, help='path to dataset')

args = parser.parse_args()

DATA_PATH = args.dataset_path
NUM_WORDS = 1000
MAX_LEN = 20
BATCH_SIZE = 32

if __name__ == '__main__':
    print('Loading data')
    dataset = Dataset(DATA_PATH, num_words=NUM_WORDS, max_len=MAX_LEN)
    dataloader = Dataloader(dataset, BATCH_SIZE)

    # TODO: fit model
