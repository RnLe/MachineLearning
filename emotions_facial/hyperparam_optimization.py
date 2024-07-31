import argparse
import optuna
import densenet
from tensorflow.keras import optimizers
import tensorflow as tf
import numpy as np
from util import evaluate, load_data, create_class_weights, plot_history
import dataloader


def gaussian_peak(min_value, max_value, length, sharpness=0.01):
    """
    Example
    -------
    min_value = 4
    max_value = 32
    length = 7
    gaussian_peak(min_value, max_value, length)
    """
    offset = length // 2 + length // 5
    x = np.linspace(0, length - 1, length)
    peak_position = (max_value - min_value) / 2 + min_value
    peak = np.exp(-sharpness * ((x - offset) ** 2))
    scaled_peak = (peak - np.min(peak)) / (np.max(peak) - np.min(peak))
    scaled_peak = (max_value - min_value) * scaled_peak + min_value
    return np.round(scaled_peak).astype(int)


def create_study(args):
    # optimization arguments
    plot = args.plot
    epochs = args.epochs
    file = args.file
    num_trials = args.num_trials

    # model & data
    input_shape = (48, 48, 1)
    train_dir = "dataset/train"
    test_dir = "dataset/test"
    classes = sorted(["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"])
    greyscale = True
    augmentation = None
    train_ds, test_ds = dataloader.load(
        train_dir, test_dir, 7, OneChannelOnly=greyscale, augmentations=augmentation
    )
    _, _, y_train_encoded = load_data(train_dir, classes, greyscale=greyscale)
    class_weights_dict = create_class_weights(y_train_encoded)

    # Load test data
    x_test, y_test, y_test_encoded = load_data(test_dir, classes, greyscale=greyscale)

    def objective(trial):
        # Suggest the number of dense blocks
        num_blocks = trial.suggest_int("num_blocks", 2, 10)

        # Suggest the max and min layers per block
        min_layers = trial.suggest_int("min_layers", 4, 8)
        max_layers = trial.suggest_int("max_layers", 16, 32)
        num_layers_per_block = gaussian_peak(min_layers, max_layers, num_blocks)
        print("XXX", num_layers_per_block)
        # Suggest other parameters
        growth_rate = trial.suggest_categorical("growth_rate", [16, 32, 48, 64])
        reduction = trial.suggest_uniform("reduction", 0.3, 0.7)
        dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5)
        l2_regularization = trial.suggest_loguniform("l2_regularization", 1e-6, 1e-3)

        # Define the model using the suggested parameters
        args = dict(
            input_shape=input_shape,
            num_blocks=num_blocks,
            num_layers_per_block=num_layers_per_block,
            growth_rate=growth_rate,
            reduction=reduction,
            num_classes=len(classes),
            dropout_rate=dropout_rate,
            l2_regularization=l2_regularization,
        )

        model = densenet.DenseNet(**args)
        model.build((None, *input_shape))
        model.compile(
            optimizer=optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        # Callback functions
        metric = "val_accuracy"
        mode = "max"
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=metric, patience=10, restore_best_weights=True, mode=mode
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=metric, factor=0.5, patience=5, min_lr=1e-7, mode=mode
        )
        # training
        history = model.fit(
            train_ds,
            class_weight=class_weights_dict,
            epochs=epochs,
            validation_data=test_ds,
            callbacks=[reduce_lr, early_stopping],
        )
        # evaluation (optional!)
        if plot:
            plot_history(history)
            evaluate(model, x_test, y_test, y_test_encoded, classes)
        # return loss (necessary)
        loss, accuracy = model.evaluate(x_test, y_test_encoded)
        return loss

    study = optuna.create_study(
        direction="minimize",
        study_name="custom-densenet",
        storage=f"sqlite:///{file}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=num_trials)
    return study


def main():
    from optparse import OptionParser

    parser = OptionParser(description="Run hyperparameter optimization for DenseNet.")
    default_num_trials = 200
    parser.add_option(
        "-n",
        "--num-trials",
        dest="num_trials",
        default=default_num_trials,
        type=int,
        help="Number of trials for the optimization.",
    )
    default_epochs = 15
    parser.add_option(
        "-e",
        "--epochs",
        dest="epochs",
        default=default_epochs,
        type=int,
        help=f"Number of epochs to train each model (default: {default_epochs}).",
    )
    default_file = "optuna.db"
    parser.add_option(
        "-f",
        "--file",
        dest="file",
        type=str,
        default=default_file,
        help=f"File name to use as storage for evaluations (default: {default_file}).",
    )
    parser.add_option(
        "-v",
        "--verbose",
        dest="plot",
        action="store_true",
        default=False,
        help=f"Plot evaluation of each model after every model training & evaluation step (default: False).",
    )
    opts, args = parser.parse_args()

    print(
        f"Starting hyperparameter optimization with {opts.num_trials} trials ({opts.epochs} epochs training)..."
    )
    study = create_study(opts)
    print("Optimization complete.")

    # Output the best trial
    trial = study.best_trial
    print("Best trial:")
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
