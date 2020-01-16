from models import get_deep_cross_model
import keras
import os


def fit(
        train_gen,
        val_gen,
        num_words,
        embedding_dim,
        max_len,
        matrix_similarity_function,
        exp_dir,
        early_stopping_after,
        convs_depth,
        denses_depth,
        epochs=20,
        lstm_dimension=50,
        embedding_matrix=None,
        embedding_trainable=False,
        dropout=False,
        activation="sigmoid",
        verbosity=2,
        callbacks=False
):
    graph_base_dir = os.path.join(exp_dir, "graph")
    checkpoint_base_dir = os.path.join(exp_dir, "checkpoint")
    logs_dir = os.path.join(exp_dir, "logs")
    csv_path = os.path.join(logs_dir, "log.csv")

    os.makedirs(graph_base_dir, exist_ok=True)
    os.makedirs(checkpoint_base_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    callbacks_list = []
    if callbacks:
        cb_tb = keras.callbacks.TensorBoard(
            log_dir=graph_base_dir, histogram_freq=0, write_graph=True, write_images=True
        )

        best_path = os.path.join(checkpoint_base_dir, "best.h5")

        cb_best = keras.callbacks.ModelCheckpoint(
            best_path, "val_loss", save_best_only=True
        )

        cb_stop = keras.callbacks.EarlyStopping(
            "val_accuracy",
            patience=early_stopping_after,
            mode="max",
            restore_best_weights=True,
        )

        cb_csv = keras.callbacks.CSVLogger(csv_path)
        callbacks_list = [cb_tb, cb_stop, cb_csv, cb_best]

    model = get_deep_cross_model(
        num_words + 1,
        embedding_dim,
        max_len,
        matrix_similarity_function,
        convs_depth=convs_depth,
        denses_depth=denses_depth,
        embedding_matrix=embedding_matrix,
        embedding_trainable=embedding_trainable,
        dropout=dropout,
        lstm_dimension=lstm_dimension,
        activation=activation
    )
    model.summary()
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("* TRAINING")
    model.fit(train_gen, validation_data=val_gen, epochs=epochs,
              callbacks=callbacks_list, verbose=verbosity)
    return model
