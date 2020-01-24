from dataset import get_data, get_iterators
from fitness import fit, evaluate
from utils import load_embedding, resave_w2v_model, resave_fasttext_model
import pickle

if __name__ == '__main__':
    config = {
        'embedding_path':
            './dataset/embeddings/w2v'
            '/new_w2v_title_300Epochs_1MinCount_9ContextWindow_100d'
            '.bin',
        'epochs': 30,
        'lr': 1e-03,
        'rnn_units': 200,
        'convs_filter_banks': 32,
        'denses_depth': 32,
        'similarity_type': 'dot',
        'automl_path': None
    }

    if config['automl_path']:
        with open(config['automl_path'], 'rb') as f:
            params = pickle.load(f)
        max_f1 = 0
        best_params = {}
        for param in params:
            if param[1] > max_f1:
                best_params = param[0]
                max_f1 = param[1]
        config['lr'] = best_params['lr']
        config['rnn_units'] = int(best_params['rnn_units'])
        config['convs_filter_banks'] = int(best_params['convs_filter_banks'])
        config['denses_depth'] = int(best_params['denses_depth'])
        config['similarity_type'] = "dot" if best_params['similarity_type'] == 0 else 'cosine'
        print(config)


    train_ds, valid_ds, test_ds, TEXT = get_data(
        train_path='./dataset/computers/train/computers_train_medium.json',
        valid_path='./dataset/computers/valid/computers_splitted_valid_medium.json',
        test_path='./dataset/computers/test/computers_gs.json'
    )
    train_dl, valid_dl, test_dl = get_iterators(train_ds, valid_ds, test_ds)
    load_embedding(TEXT, config['embedding_path'])

    model = fit(TEXT, train_dl, valid_dl, config=config, hidden_dim=config['convs_filter_banks'],
                lr=config['lr'], similarity='dot', loss='BCELoss', conv_depth=3,
                dense_depth=config['denses_depth'], validate_each_epoch=True)
    evaluate(model, test_dl, print_results=True)
