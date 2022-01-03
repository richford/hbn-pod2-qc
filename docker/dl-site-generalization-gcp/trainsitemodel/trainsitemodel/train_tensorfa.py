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
    GCS_BASE_PATH = (
        f"gs://{gcs_bucket}/site_generalization_jobs/{job_name}/seed_{dataset_seed}"
    )
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
        device_tfrec_dir = GCS_TFREC_DIR
    else:
        LOCAL_NIFTI_DIR = op.join(".", "nifti")
        LOCAL_TFREC_DIR = op.join(".", "tfrecs")
        os.makedirs(LOCAL_NIFTI_DIR, exist_ok=True)
        os.makedirs(LOCAL_TFREC_DIR, exist_ok=True)
        fs.get(GCS_NIFTI_DIR, LOCAL_NIFTI_DIR, recursive=True)
        fs.get(GCS_TFREC_DIR, LOCAL_TFREC_DIR, recursive=True)
        device_nifti_dir = LOCAL_NIFTI_DIR
        device_tfrec_dir = LOCAL_TFREC_DIR

    train_validate_tfrec_dir = op.join(
        device_tfrec_dir,
        "_".join(train_sites),
        f"seed-{dataset_seed}",
    )

    test_report_tfrec_dir = op.join(
        device_tfrec_dir,
        "_".join(test_sites),
    )

    n_classes = 1
    batch_size = 16
    volume_shape = (128, 128, 128, n_channels)
    block_shape = (128, 128, 128, n_channels)
    num_parallel_calls = 4

    s3_participants = pd.read_csv(
        "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv",
        sep="\t",
        usecols=["subject_id", "scan_site_id", "expert_qc_score", "xgb_qc_score"],
        index_col="subject_id",
    )

    nifti_files = [
        op.abspath(filename) for filename in glob(f"{device_nifti_dir}/*.nii.gz")
    ]
    nifti_files = [fn for fn in nifti_files if "irregularsize" not in fn]
    sub_id_pattern = re.compile("sub-[a-zA-Z0-9]*")
    subjects = [sub_id_pattern.search(s).group(0) for s in nifti_files]

    participants = pd.DataFrame(data=nifti_files, index=subjects, columns=["features"])
    participants = participants.merge(
        s3_participants, left_index=True, right_index=True, how="left"
    )
    participants.dropna(subset=["xgb_qc_score"], inplace=True)
    participants.rename(columns={"xgb_qc_score": "labels"}, inplace=True)

    # Get site inclusion indices for both the report set and
    # the remaining train/validate/test (tvt) sets
    # Take out the report set by checking for existence of expert QC scores
    split_dataframes = {
        "report": participants.loc[
            np.logical_and(
                participants["scan_site_id"].isin(test_sites),
                ~participants["expert_qc_score"].isna(),
            )
        ],
        "test": participants.loc[
            np.logical_and(
                participants["scan_site_id"].isin(test_sites),
                participants["expert_qc_score"].isna(),
            )
        ],
    }

    test_size = len(split_dataframes["test"])
    report_size = len(split_dataframes["report"])
    num_volumes = pd.read_csv(
        op.join(train_validate_tfrec_dir, "num_volumes.csv"), index_col="split"
    )

    assert test_size == num_volumes.loc["test"].to_numpy()[0]
    assert report_size == num_volumes.loc["report"].to_numpy()[0]
    train_size = num_volumes.loc["train"].to_numpy()[0]
    validate_size = num_volumes.loc["validate"].to_numpy()[0]

    train_sites_str = "data-" + "_".join(train_sites)
    test_sites_str = "data-" + "_".join(test_sites)
    split_patterns = {
        "train": op.join(
            train_validate_tfrec_dir, f"{train_sites_str}-train_shard*.tfrec"
        ),
        "validate": op.join(
            train_validate_tfrec_dir, f"{train_sites_str}-validate_shard*.tfrec"
        ),
        "test": op.join(
            train_validate_tfrec_dir, f"{test_sites_str}-test_shard*.tfrec"
        ),
        "report": op.join(
            train_validate_tfrec_dir, f"{test_sites_str}-report_shard*.tfrec"
        ),
    }

    datasets = {
        split: nobrainer.dataset.get_dataset(
            file_pattern=file_pattern,
            n_classes=n_classes,
            batch_size=batch_size,
            volume_shape=volume_shape,
            scalar_label=True,
            # block_shape=block_shape,
            augment=True,
            n_epochs=None if split == "train" else 1,
            num_parallel_calls=num_parallel_calls,
        )
        for split, file_pattern in split_patterns.items()
    }

    if compute_volume_numbers:
        n_train_volumes = sum(len(y) for _, y in datasets["train"].as_numpy_iterator())
        n_validate_volumes = sum(
            len(y) for _, y in datasets["validate"].as_numpy_iterator()
        )
        n_test_volumes = sum(len(y) for _, y in datasets["test"].as_numpy_iterator())
        n_report_volumes = sum(
            len(y) for _, y in datasets["report"].as_numpy_iterator()
        )
    else:
        n_train_volumes = train_size // batch_size * batch_size
        n_validate_volumes = validate_size // batch_size * batch_size
        n_test_volumes = test_size // batch_size * batch_size
        n_report_volumes = report_size // batch_size * batch_size

    print("n_train_volumes: ", n_train_volumes)
    print("n_validate_volumes: ", n_validate_volumes)
    print("n_test_volumes: ", n_test_volumes)
    print("n_report_volumes: ", n_report_volumes)

    steps_per_epoch = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_train_volumes,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_size,
    )

    validation_steps = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_validate_volumes,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_size,
    )

    test_steps = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_test_volumes,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_size,
    )

    report_steps = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_report_volumes,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_size,
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
        batch_size=batch_size,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=datasets["validate"],
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    model.save(SAVED_MODEL_DIR)
    model.evaluate(datasets["test"])

    fs.put(LOCAL_MODEL_CHECKPOINT_DIR, MODEL_CHECKPOINT_DIR, recursive=True)
    fs.put(LOCAL_CSV_LOGGER_FILEPATH, CSV_LOGGER_FILEPATH)

    print("Predicting the report set.")
    y_hat_test = model.predict(datasets["test"])
    y_test = np.concatenate([y.numpy() for _, y in datasets["test"]])
    df_test = pd.DataFrame(
        dict(
            y_true=np.squeeze(y_test),
            y_prob=np.squeeze(y_hat_test),
        ),
        index=split_dataframes["test"].index,
    )
    df_test.to_csv(op.join(LOCAL_PREDICTION_OUTPUT_DIR, f"test_set_{dataset_seed}.csv"))

    print("Predicting the report set.")
    y_hat_report = model.predict(datasets["report"])
    y_report = np.concatenate([y.numpy() for _, y in datasets["report"]])
    df_report = pd.DataFrame(
        dict(
            y_true=np.squeeze(y_report),
            y_prob=np.squeeze(y_hat_report),
        ),
        index=split_dataframes["report"].index,
    )
    df_report.to_csv(
        op.join(LOCAL_PREDICTION_OUTPUT_DIR, f"report_set_{dataset_seed}.csv")
    )
    fs.put(LOCAL_PREDICTION_OUTPUT_DIR, PREDICTIONS_DIR, recursive=True)


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