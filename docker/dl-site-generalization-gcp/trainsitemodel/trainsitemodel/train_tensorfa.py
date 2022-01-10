import argparse
import gcsfs
import nobrainer
import numpy as np
import os
import os.path as op
import pandas as pd
import re
import tensorflow as tf

from glob import glob


def build_multi_input_model(
    width=128,
    height=128,
    depth=128,
    n_image_channels=3,
    n_qc_metrics=31,
    neglect_qc=False,
):
    """Build a multi-input model that concatenates a 3D convolutional neural
    network model and a dense model for QC inputs.
    """

    inputs = tf.keras.Input((width, height, depth, n_image_channels + 1))

    image_input = tf.keras.layers.Lambda(
        lambda a: a[:, :, :, :, :n_image_channels], name="image_input"
    )(inputs)

    qc_metric_input = tf.keras.layers.Lambda(
        lambda a: a[:, 0, 0, :n_qc_metrics, -1], name="qc_metric_input"
    )(inputs)

    conv_net = build_model(
        width=width,
        height=height,
        depth=depth,
        n_channels=n_image_channels,
        include_top=False,
    )(image_input)

    x = tf.keras.layers.Flatten()(qc_metric_input)

    if neglect_qc:
        x = tf.zeros_like(x)

    x = tf.keras.layers.concatenate([conv_net, x])
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = tf.keras.Model(inputs, outputs, name="multi_input_3dcnn")
    return model


def build_model(width=128, height=128, depth=128, n_channels=3, include_top=True):
    """Build a 3D convolutional neural network model."""

    inputs = tf.keras.Input((width, height, depth, n_channels))

    x = tf.keras.layers.Conv3D(
        filters=64, kernel_size=3, padding="same", activation="relu"
    )(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(
        filters=64, kernel_size=3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(
        filters=128, kernel_size=3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(
        filters=256, kernel_size=3, padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)

    if not include_top:
        return tf.keras.Model(inputs, x)
    else:
        x = tf.keras.layers.Dense(units=512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

        # Define the model.
        model = tf.keras.Model(inputs, outputs, name="3dcnn")
        return model


def label_y(x, y):
    return x, tf.cast(y >= 0.5, dtype=tf.int32)


def main(
    gcs_bucket,
    job_name,
    train_sites,
    test_sites,
    n_epochs=5,
    n_channels=5,
    dataset_name="b0-tensorfa-dwiqc",
    dataset_seed=0,
    model_loss="binary_crossentropy",
    compute_volume_numbers=False,
    neglect_qc=False,
):
    print("Setting gpu thread mode to gpu_private.")
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

    print("Configuring distribution strategy")
    use_tpu = False

    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="")
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

        use_tpu = True
        print("TPU detected.")
        print("All devices: ", tf.config.list_logical_devices("TPU"))
    except ValueError:
        strategy = tf.distribute.MirroredStrategy()
        print("GPUs detected.")
        print("Number of accelerators: ", strategy.num_replicas_in_sync)

    # Train using mixed-precision policy
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    scope = strategy.scope()

    if model_loss not in ["binary_crossentropy", "mean_squared_error"]:
        raise ValueError(
            "model_loss must be either binary_crossentropy or mean_squared_error"
        )

    # Setting location were training logs and checkpoints will be stored
    GCS_BASE_PATH = f"gs://{gcs_bucket}/site_generalization_jobs/{job_name}/train_{'_'.join(train_sites)}_test_{'_'.join(test_sites)}/seed_{dataset_seed}"
    TENSORBOARD_LOGS_DIR = op.join(GCS_BASE_PATH, "logs")
    MODEL_CHECKPOINT_DIR = op.join(GCS_BASE_PATH, "checkpoints")
    CSV_LOGGER_FILEPATH = op.join(
        GCS_BASE_PATH, "checkpoints", f"training_{dataset_seed}.csv"
    )
    SAVED_MODEL_DIR = op.join(GCS_BASE_PATH, "saved_model")
    PREDICTIONS_DIR = op.join(GCS_BASE_PATH, "predictions")

    fs = gcsfs.GCSFileSystem()
    fs.touch(CSV_LOGGER_FILEPATH)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("saved_model", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    LOCAL_MODEL_CHECKPOINT_DIR = "checkpoints"
    LOCAL_PREDICTION_OUTPUT_DIR = "predictions"
    LOCAL_MODEL_CHECKPOINT_FILEPATH = op.join(
        LOCAL_MODEL_CHECKPOINT_DIR, "model_{epoch:03d}-{val_loss:.2f}.h5"
    )
    LOCAL_CSV_LOGGER_FILEPATH = op.join("checkpoints", f"training_{dataset_seed}.csv")

    # Specify the datasets on GCP storage
    GCS_DATA_PATH = f"gs://{gcs_bucket}"
    GCS_NIFTI_DIR = op.join(GCS_DATA_PATH, "nifti")
    GCS_TFREC_DIR = op.join(GCS_DATA_PATH, "tfrecs")

    if use_tpu:
        device_nifti_dir = GCS_NIFTI_DIR
        train_tfrec_dir = op.join(
            GCS_TFREC_DIR, "_".join(train_sites), f"seed-{dataset_seed}"
        )
        test_tfrec_dir = op.join(GCS_TFREC_DIR, "_".join(test_sites))
    else:
        device_nifti_dir = op.join(".", "nifti")
        device_tfrec_dir = op.join(".", "tfrecs")
        train_tfrec_dir = op.join(
            device_tfrec_dir, "_".join(train_sites), f"seed-{dataset_seed}"
        )
        test_tfrec_dir = op.join(device_tfrec_dir, "_".join(test_sites))
        os.makedirs(device_nifti_dir, exist_ok=True)
        os.makedirs(train_tfrec_dir, exist_ok=True)
        os.makedirs(test_tfrec_dir, exist_ok=True)
        fs.get(GCS_NIFTI_DIR, device_nifti_dir, recursive=True)
        fs.get(
            "/".join([GCS_TFREC_DIR, "_".join(train_sites), f"seed-{dataset_seed}"]),
            train_tfrec_dir,
            recursive=True,
        )
        gcs_test_dir = "/".join([GCS_TFREC_DIR, "_".join(test_sites)])
        test_files = fs.glob("/".join([gcs_test_dir, "data-*.tfrec"]))
        for fname in test_files:
            fs.get(fname, op.join(test_tfrec_dir, op.basename(fname)))

    n_classes = 1
    volume_shape = (128, 128, 128, n_channels)
    block_shape = (128, 128, 128, n_channels)
    num_parallel_calls = 4

    train_volumes = pd.read_csv(
        op.join(train_tfrec_dir, "num_volumes.csv"), index_col="split"
    )
    test_volumes = pd.read_csv(
        "/".join([gcs_test_dir, "seed-0", "num_volumes.csv"]), index_col="split"
    )
    train_subjects = pd.read_csv(
        op.join(train_tfrec_dir, "train-subjects.csv"), index_col=0
    )
    validate_subjects = pd.read_csv(
        op.join(train_tfrec_dir, "validate-subjects.csv"), index_col=0
    )
    test_subjects = pd.read_csv(op.join(gcs_test_dir, "test-subjects.csv"), index_col=0)
    report_subjects = pd.read_csv(
        op.join(gcs_test_dir, "report-subjects.csv"), index_col=0
    )

    test_size = test_volumes.loc["test"].to_numpy()[0]
    report_size = test_volumes.loc["report"].to_numpy()[0]
    train_size = train_volumes.loc["train"].to_numpy()[0]
    validate_size = train_volumes.loc["validate"].to_numpy()[0]

    assert test_size == len(test_subjects)
    assert report_size == len(report_subjects)
    assert train_size == len(train_subjects)
    assert validate_size == len(validate_subjects)

    # Set the batch size to divisors of the test and report datasets
    batch_sizes = {}
    for split, _size in zip(
        ["test", "report"],
        [test_size, report_size],
    ):
        for candidate_size in range(1, 32):
            if _size % candidate_size == 0:
                batch_sizes[split] = candidate_size

    # For the train and validate datasets, it is not as important that the batch
    # size divides the total number of subjects
    batch_sizes["train"] = 16
    batch_sizes["validate"] = 16
    batch_sizes["test"] = 1
    batch_sizes["report"] = 1

    train_sites_str = "data-" + "_".join(train_sites)
    test_sites_str = "data-" + "_".join(test_sites)
    split_patterns = {
        "train": op.join(train_tfrec_dir, f"{train_sites_str}-train_shard*.tfrec"),
        "validate": op.join(
            train_tfrec_dir, f"{train_sites_str}-validate_shard*.tfrec"
        ),
        "test": op.join(test_tfrec_dir, f"{test_sites_str}-test_shard*.tfrec"),
        "report": op.join(test_tfrec_dir, f"{test_sites_str}-report_shard*.tfrec"),
    }

    datasets = {
        split: nobrainer.dataset.get_dataset(
            file_pattern=file_pattern,
            n_classes=n_classes,
            batch_size=batch_sizes[split],
            volume_shape=volume_shape,
            scalar_label=True,
            # block_shape=block_shape,
            augment=True,
            n_epochs=None if split == "train" else 1,
            num_parallel_calls=num_parallel_calls,
        )
        for split, file_pattern in split_patterns.items()
    }

    n_train_volumes = train_size // batch_sizes["train"] * batch_sizes["train"]
    n_validate_volumes = (
        validate_size // batch_sizes["validate"] * batch_sizes["validate"]
    )
    n_test_volumes = test_size // batch_sizes["test"] * batch_sizes["test"]
    n_report_volumes = report_size // batch_sizes["report"] * batch_sizes["report"]

    if compute_volume_numbers:
        print("Computing number of test volumes...")
        n_test_volumes = sum(len(y) for _, y in datasets["test"].as_numpy_iterator())
        print("Computing number of report volumes...")
        n_report_volumes = sum(
            len(y) for _, y in datasets["report"].as_numpy_iterator()
        )

    print("n_train_volumes: ", n_train_volumes)
    print("n_validate_volumes: ", n_validate_volumes)
    print("n_test_volumes: ", n_test_volumes)
    print("n_report_volumes: ", n_report_volumes)

    for split, batch_size in batch_sizes.items():
        print(f"{split} batch size: {batch_size}")

    steps_per_epoch = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_train_volumes,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_sizes["train"],
    )

    validation_steps = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_validate_volumes,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_sizes["validate"],
    )

    test_steps = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_test_volumes,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_sizes["test"],
    )

    report_steps = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_report_volumes,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_sizes["report"],
    )

    if use_tpu:
        compile_kwargs = {"steps_per_execution": steps_per_epoch // 2}
    else:
        compile_kwargs = {}

    with scope:
        # Set learning rate schedule
        initial_learning_rate = 0.0001

        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate,
        #     decay_steps=steps_per_epoch * 2,
        #     decay_rate=0.94,
        #     staircase=True,
        # )
        # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

        if model_loss == "mean_squared_error":
            # Define binary accuracy and AUC metrics when y_true is a probability
            def binary_accuracy_with_prob(y_prob, y_pred):
                y_true = tf.cast(tf.round(y_prob), tf.int32)
                m = tf.keras.metrics.BinaryAccuracy()
                m.update_state(y_true, y_pred)
                return m.result().numpy()

            model = build_multi_input_model(
                n_image_channels=n_channels - 1, neglect_qc=neglect_qc
            )

            model.compile(
                loss=model_loss,
                optimizer=optimizer,
                metrics=[
                    model_loss,
                    binary_accuracy_with_prob,
                ],
                **compile_kwargs,
                # run_eagerly=True,
            )
        elif model_loss == "binary_crossentropy":
            datasets["train"] = datasets["train"].map(
                label_y, num_parallel_calls=num_parallel_calls
            )
            datasets["validate"] = datasets["validate"].map(
                label_y, num_parallel_calls=num_parallel_calls
            )
            datasets["test"] = datasets["test"].map(
                label_y, num_parallel_calls=num_parallel_calls
            )
            datasets["report"] = datasets["report"].map(
                label_y, num_parallel_calls=num_parallel_calls
            )

            model = build_multi_input_model(
                n_image_channels=n_channels - 1, neglect_qc=neglect_qc
            )

            model.compile(
                loss=model_loss,
                optimizer=optimizer,
                metrics=[
                    model_loss,
                    "accuracy",
                    "AUC",
                ],
                **compile_kwargs,
                # run_eagerly=True,
            )

    print(model.summary())

    csv_logger = tf.keras.callbacks.CSVLogger(
        LOCAL_CSV_LOGGER_FILEPATH,
        separator=",",
        append=True,
    )

    best_model = tf.keras.callbacks.ModelCheckpoint(
        LOCAL_MODEL_CHECKPOINT_FILEPATH,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
    )

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS_DIR)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=20,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
    )

    # Configure Tensorboard logs
    callbacks = [
        tensorboard,
        best_model,
        csv_logger,
        early_stop,
        reduce_lr,
    ]

    print("Training the model.")

    # Train the model, doing validation at the end of each epoch
    _ = model.fit(
        datasets["train"],
        batch_size=batch_sizes["train"],
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=datasets["validate"],
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    model.save(SAVED_MODEL_DIR)

    fs.put(LOCAL_MODEL_CHECKPOINT_DIR, MODEL_CHECKPOINT_DIR, recursive=True)
    fs.put(LOCAL_CSV_LOGGER_FILEPATH, CSV_LOGGER_FILEPATH)

    print("Predicting the report set.")
    y_hat_report = model.predict(datasets["report"], steps=report_steps)
    y_report = np.concatenate([y.numpy() for _, y in datasets["report"]])
    df_report = pd.DataFrame(
        dict(
            y_true=np.squeeze(y_report),
            y_prob=np.squeeze(y_hat_report),
        ),
        index=report_subjects.index,
    )
    df_report["y_from_tsv"] = report_subjects["labels"]
    df_report["train_sites"] = " + ".join(train_sites)
    df_report["test_sites"] = " + ".join(test_sites)
    df_report["seed"] = dataset_seed
    df_report.to_csv(
        op.join(LOCAL_PREDICTION_OUTPUT_DIR, f"report_set_{dataset_seed}.csv")
    )

    fs.put(LOCAL_PREDICTION_OUTPUT_DIR, PREDICTIONS_DIR, recursive=True)

    print("Predicting the test set.")
    y_hat_test = model.predict(datasets["test"], steps=test_steps)
    y_test = np.concatenate([y.numpy() for _, y in datasets["test"]])
    df_test = pd.DataFrame(
        dict(
            y_true=np.squeeze(y_test),
            y_prob=np.squeeze(y_hat_test),
        ),
        index=test_subjects.index,
    )
    df_test["y_from_tsv"] = test_subjects["labels"]
    df_test["train_sites"] = " + ".join(train_sites)
    df_test["test_sites"] = " + ".join(test_sites)
    df_test["seed"] = dataset_seed
    df_test.to_csv(op.join(LOCAL_PREDICTION_OUTPUT_DIR, f"test_set_{dataset_seed}.csv"))

    fs.put(LOCAL_PREDICTION_OUTPUT_DIR, PREDICTIONS_DIR, recursive=True)

    model.evaluate(datasets["test"], steps=test_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gcs_bucket",
        type=str,
        help=(
            "The name of the gcs bucket that will contain the saved models, "
            "checkpoints, etc."
        ),
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help=(
            "The job name for this job. Results will be stored in a "
            "'subdirectory' with this name inside the GCS bucket."
        ),
    )
    parser.add_argument(
        "--train_site",
        action="append",
        help="Add a site to training/validate set.",
        required=True,
    )
    parser.add_argument(
        "--test_site",
        action="append",
        help="Add a site to the test/report set.",
        required=True,
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        help="The number of training epochs.",
        default=5,
    )
    parser.add_argument(
        "--n_channels",
        type=int,
        help="The number of channels in the data.",
        default=5,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The name of the dataset in the tfrecs folder of the GCS bucket.",
        default="b0-tensorfa-dwiqc",
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        help="The seed for the dataset",
        default=0,
    )
    parser.add_argument(
        "--model_loss",
        type=str,
        help="The model loss function",
        default="binary_crossentropy",
    )
    parser.add_argument(
        "--compute-volume-numbers", dest="compute_volume_numbers", action="store_true"
    )
    parser.add_argument(
        "--no-compute-volume-numbers",
        dest="compute_volume_numbers",
        action="store_false",
    )
    parser.add_argument(
        "--neglect-qc",
        dest="neglect_qc",
        action="store_true",
    )
    parser.set_defaults(compute_volume_numbers=False)

    args = parser.parse_args()

    main(
        gcs_bucket=args.gcs_bucket,
        job_name=args.job_name,
        train_sites=args.train_site,
        test_sites=args.test_site,
        n_epochs=args.n_epochs,
        n_channels=args.n_channels,
        dataset_name=args.dataset_name,
        dataset_seed=args.dataset_seed,
        model_loss=args.model_loss,
        compute_volume_numbers=args.compute_volume_numbers,
        neglect_qc=args.neglect_qc,
    )