import json
import argparse
import numpy as np
from pandas import pandas
from utils import parse_content_line

parser = argparse.ArgumentParser(description="Split dataset")
parser.add_argument(
    "--train-path",
    type=str,
    help="path to train dataset",
    default="./dataset/computers/train/computers_train_medium.json",
)
parser.add_argument(
    "--validation-path",
    type=str,
    help="path to validation dataset",
    default="./dataset/computers/valid/computers_valid_medium.csv",
)
parser.add_argument(
    "--splitted-train-path",
    type=str,
    help="path to write splitted train dataset",
    default="./dataset/computers/train/computers_splitted_train_medium.json",
)
parser.add_argument(
    "--splitted-validation-path",
    type=str,
    help="path to write splitted validation dataset",
    default="./dataset/computers/valid/computers_splitted_valid_medium.json",
)

args = parser.parse_args()

attributes = list(json.loads(open(args.train_path, "r").readline()).keys())

train = []
for i, x in enumerate(open(args.train_path, "r").readlines()):
    try:
        item = parse_content_line(x, attributes=attributes, label=0)
        train.append(item)
    except:
        print("Lost data at line {}".format(i))
train = np.concatenate(train, axis=0)
train = pandas.DataFrame(data=train, columns=attributes)

valid = pandas.read_csv(args.validation_path)
merged = pandas.merge(train, valid, on=["pair_id"], how="left", indicator=True)
valid = merged[merged["_merge"] == "both"].drop("_merge", axis=1)
train = merged[merged["_merge"] == "left_only"].drop("_merge", axis=1)

with open(args.splitted_train_path, "w") as f:
    for item in train.to_dict(orient="records"):
        f.write(json.dumps(item) + "\n")
    f.flush()
    f.close()

with open(args.splitted_validation_path, "w") as f:
    for item in valid.to_dict(orient="records"):
        f.write(json.dumps(item) + "\n")
    f.flush()
    f.close()
