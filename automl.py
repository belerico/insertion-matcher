import numpy as np
import functools

from pyGPGO.covfunc import squaredExponential
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO
from pyGPGO.acquisition import Acquisition

from dataset import get_data, get_iterators
from fitness import fit, evaluate
from utils import load_embedding


def myFirstRun(self, init_rand_configs=None, n_eval=3):
    """
    Performs initial evaluations before fitting GP.

    Parameters
    ----------
    init_rand_configs: list
        Initial random configurations
    n_eval: int
        Number of initial evaluations to perform. Default is 3.

    """
    if init_rand_configs is None:
        self.X = np.empty((n_eval, len(self.parameter_key)))
        self.y = np.empty((n_eval,))
        for i in range(n_eval):
            s_param = self._sampleParam()
            s_param_val = list(s_param.values())
            self.X[i] = s_param_val
            self.y[i] = self.f(**s_param)
    else:
        self.X = np.empty((len(init_rand_configs), len(init_rand_configs[0])))
        self.y = np.empty((len(init_rand_configs),))
        self.init_evals = len(self.y)
        for i in range(len(init_rand_configs)):
            self.X[i] = list(init_rand_configs[i].values())
            self.y[i] = self.f(**init_rand_configs[i])
    self.GP.fit(self.X, self.y)
    self.tau = np.max(self.y)
    if init_rand_configs is None:
        self.history.append([init_rand_configs[np.argmax(self.y)], self.tau])
    else:
        idx_max_param = np.argmax(self.y)
        self.history.append(
            [
                {
                    key: self.X[idx_max_param, idx]
                    for idx, key in enumerate(self.parameter_key)
                },
                self.GP.y[-1],
                self.tau,
            ]
        )


def myUpdateGP(self):
    """
    Updates the internal model with the next acquired point and its evaluation.
    """
    kw = {
        param: int(self.best[i])
        if self.parameter_type[i] == "int"
        else float(self.best[i])
        for i, param in enumerate(self.parameter_key)
    }
    f_new = self.f(**kw)
    self.GP.update(np.atleast_2d(self.best), np.atleast_1d(f_new))
    self.tau = np.max(self.GP.y)
    self.history.append([kw, self.GP.y[-1], self.tau])


def get_fitness_for_automl(config):
    train_ds, valid_ds, test_ds, TEXT = get_data(
        config["train_path"], config["valid_path"], config["test_path"],
    )
    train_dl, valid_dl, test_dl = get_iterators(train_ds, valid_ds, test_ds)
    load_embedding(TEXT, config["embedding_path"])

    def fitness(
        lr,
        rnn_units,
        convs_filter_banks,
        convs_kernel_size,
        dense_depth1,
        dense_depth2,
        similarity_type,
    ):
        similarity_type = "dot" if similarity_type == 0 else "cosine"

        model = fit(
            TEXT,
            train_dl,
            valid_dl,
            config=config,
            hidden_dim=rnn_units,
            conv_depth=convs_filter_banks,
            kernel_size=convs_kernel_size,
            dense_depth1=dense_depth1,
            dense_depth2=dense_depth2,
            lr=lr,
            similarity=similarity_type,
            loss="CrossEntropyLoss",
            validate_each_epoch=True,
        )

        result = evaluate(model, valid_dl, print_results=False)
        return result

    return fitness


if __name__ == "__main__":
    config = {
        "expname": "w2v_10Epochs_100d_CrossEntropy_BothDenses",
        "train_path": "./dataset/computers/train/computers_splitted_train_medium.json",
        "valid_path": "./dataset/computers/valid/computers_splitted_valid_medium.json",
        "test_path": "./dataset/computers/test/computers_gs.json",
        "embedding_path": "./dataset/embeddings/w2v/w2v_title_300Epochs_1MinCount_9ContextWindow_100d"
        ".txt",
        "epochs": 10,
    }

    # ### ExpectedImprovement

    furtherEvaluations = 10

    param = {
        "lr": ("cont", [1e-6, 1e-3]),
        "rnn_units": ("int", [50, 251]),
        "convs_filter_banks": ("int", [4, 65]),
        "convs_kernel_size": ("int", [2, 4]),
        "dense_depth1": ("int", [16, 129]),
        "dense_depth2": ("int", [16, 129]),
        "similarity_type": ("int", [0, 2]),
    }

    init_rand_configs = [
        {
            "lr": 1e-3,
            "rnn_units": 100,
            "convs_filter_banks": 32,
            "convs_kernel_size": 3,
            "dense_depth1": 32,
            "dense_depth2": 16,
            "similarity_type": 0,
        }
    ]

    # creating a GP surrogate model with a Squared Exponantial covariance function,
    # aka kernel
    sexp = squaredExponential()
    sur_model = GaussianProcess(sexp)
    fitness = get_fitness_for_automl(config)
    # setting the acquisition function
    acq = Acquisition(mode="ExpectedImprovement")

    # creating an object Bayesian Optimization
    bo_step1_expected = GPGO(sur_model, acq, fitness, param, n_jobs=1)
    bo_step1_expected._firstRun = functools.partial(myFirstRun, bo_step1_expected)
    bo_step1_expected.updateGP = functools.partial(myUpdateGP, bo_step1_expected)
    bo_step1_expected._firstRun(init_rand_configs=init_rand_configs)
    bo_step1_expected.logger._printInit(bo_step1_expected)

    bo_step1_expected.run(furtherEvaluations, resume=True)

    import pickle

    with open(f'data/exps/{config["expname"]}.pickle', "wb") as handle:
        pickle.dump(bo_step1_expected.history, handle, protocol=pickle.HIGHEST_PROTOCOL)
