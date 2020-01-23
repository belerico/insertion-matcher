from dataset import get_data, get_iterators
from fitness import fit, evaluate
from utils import load_embedding

if __name__ == '__main__':
    config = {
        'embedding_dim': 100,
        'embedding_path':
            './dataset/embeddings/w2v/new_w2v_title_300Epochs_1MinCount_9ContextWindow_100d.bin',
        'epochs': 1
    }
    train_ds, valid_ds, test_ds, TEXT = get_data(
        train_path='./dataset/computers/train/computers_splitted_train_medium.json',
        valid_path='./dataset/computers/valid/computers_splitted_valid_medium.json',
        test_path='./dataset/computers/test/computers_gs.json'
    )
    train_dl, valid_dl, test_dl = get_iterators(train_ds, valid_ds, test_ds)
    load_embedding(TEXT, config['embedding_path'])

    model = fit(TEXT, train_dl, valid_dl, config=config, hidden_dim=200, lr=1e-3,
                loss='BCELoss', validate_each_epoch=True)
    evaluate(model, test_dl)
