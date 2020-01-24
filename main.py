from dataset import get_data, get_iterators
from fitness import fit, evaluate
from utils import load_embedding
import pickle

if __name__ == "__main__":
    config = {
        'embedding_path':
            './data/embeddings/fasttext/fasttext_title_300Epochs_1MinCount_9ContextWindow_150d'
            '.txt',
        'epochs': 30,
        'lr': 1e-03,
        'rnn_units': 100,
        'convs_filter_banks': 32,
        'denses_depth1': 32,
        'denses_depth2': 16,
        'similarity_type': 'dot',
        'automl_path': None,
        'rnn_step': 50,
        'conv_step': 8,
        'dense_step': 16
    }

    if config["automl_path"]:
        with open(config["automl_path"], "rb") as f:
            params = pickle.load(f)
        max_f1 = 0
        best_params = {}
        for param in params:
            if param[1] > max_f1:
                best_params = param[0]
                max_f1 = param[1]

        config["lr"] = best_params["lr"]
        config["rnn_units"] = int(best_params["rnn_units"]) * config['rnn_step']
        config["convs_filter_banks"] = int(best_params["convs_filter_banks"]) * config['conv_step']
        config["denses_depth"] = int(best_params["denses_depth"]) * config['dense_step']
        config["similarity_type"] = (
            "dot" if best_params["similarity_type"] == 0 else "cosine"
        )
        print(config)

    train_ds, valid_ds, test_ds, TEXT = get_data(
        train_path='./data/computers/train/computers_train_large.json',
        valid_path='./data/computers/valid/computers_splitted_valid_medium.json',
        test_path='./data/computers/test/computers_gs.json'
    )
    train_dl, valid_dl, test_dl = get_iterators(train_ds, valid_ds, test_ds)
    load_embedding(TEXT, config["embedding_path"])

    model = fit(
        TEXT,
        train_dl,
        valid_dl,
        config=config,
        conv_depth=config["convs_filter_banks"],
        dense_depth1=config["denses_depth1"],
        dense_depth2=config["denses_depth2"],
        hidden_dim=config["rnn_units"],
        lr=config["lr"],
        similarity="dot",
        loss="CrossEntropyLoss",
        validate_each_epoch=True,
    )
    evaluate(model, test_dl, print_results=True)
