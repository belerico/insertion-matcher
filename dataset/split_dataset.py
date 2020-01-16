import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Split dataset")
parser.add_argument(
    "--dataset-path",
    type=str,
    help="path to dataset",
    default="./dataset/all_train_small.json")
parser.add_argument(
    "--split",
    type=float,
    help="split percentage",
    default=.8
)

args = parser.parse_args()
data_path = args.dataset_path
split = args.split

lines = open(data_path, "r").readlines()
indexes = np.arange(len(lines))
np.random.shuffle(indexes)

train_test_split = int(len(lines) * split)

train_indexes = indexes[:train_test_split]
test_indexes = indexes[train_test_split:]
with open("data/train.json", "w") as train_dataset:
    for i in train_indexes:
        line = lines[i]
        train_dataset.write(line + '\n')

with open("data/test.json", "w") as test_dataset:
    for i in test_indexes:
        line = lines[i]
        test_dataset.write(line + '\n')
