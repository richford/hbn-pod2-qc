import argparse
import gcsfs
import nobrainer
import numpy as np
import os
import os.path as op
import pandas as pd
import tensorflow as tf


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


def get_site_datasets(
    device_dataset_dir,
    n_classes,
    batch_size,
    volume_shape,
    num_parallel_calls=2,
):
    dataset_all = nobrainer.dataset.get_dataset(
        file_pattern=op.join(device_dataset_dir, "data-all_shard*.tfrec"),
        n_classes=n_classes,
        batch_size=batch_size,
        volume_shape=volume_shape,
        scalar_label=True,
        # block_shape=block_shape,
        augment=True,
        n_epochs=1,
        num_parallel_calls=num_parallel_calls,
    )
    filepaths = pd.read_csv(op.join(device_dataset_dir, "filepaths.csv"))
    filepaths.columns = ["subject_id", "nifti_path", "xgb_qc"]
    filepaths.set_index("subject_id", inplace=True, drop=True)

    s3_participants = pd.read_csv(
        "s3://fcp-indi/data/Projects/HBN/BIDS_curated/derivatives/qsiprep/participants.tsv",
        sep="\t",
        usecols=["subject_id", "scan_site_id", "expert_qc_score"],
        index_col="subject_id",
    )

    participants = filepaths.merge(
        s3_participants, how="left", left_index=True, right_index=True
    )

    # Get site inclusion indices for both the report set and
    # the remaining train/validate/test (tvt) sets
    # Take out the report set by checking for existence of expert QC scores
    sites = ["RU", "CBIC", "CUNY"]
    site_indices = {
        "report": {
            site: np.where(
                np.logical_and(
                    np.logical_and(
                        participants["scan_site_id"] == site,
                        ~participants["expert_qc_score"].isna(),
                    ),
                    ~participants["xgb_qc"].isna(),
                )
            )[0]
            for site in sites
        },
        "tvt": {
            site: np.where(
                np.logical_and(
                    np.logical_and(
                        participants["scan_site_id"] == site,
                        participants["expert_qc_score"].isna(),
                    ),
                    ~participants["xgb_qc"].isna(),
                )
            )[0]
            for site in sites
        },
    }

    def sub_match(target_indices):
        def is_sub(idx, tensor):
            return tf.math.reduce_any(tf.math.equal(idx, target_indices))

        return is_sub

    site_datasets = {
        dset: {
            site: {
                "indices": idx,
                "subject_ids": participants.iloc[idx].index.tolist(),
                "dataset": dataset_all.enumerate()
                .filter(sub_match(idx))
                .map(lambda i, x: x)
                .unbatch(),
            }
            for site, idx in site_indices.items()
        }
        for dset, site_indices in site_indices.items()
    }

    return site_datasets


def split_datasets(
    site_datasets,
    train_sites_in,
    test_sites_in,
    batch_size,
    n_epochs,
    seed,
    train_fraction=0.8,
):
    train_sites = train_sites_in.copy()
    test_sites = test_sites_in.copy()

    _train_site = train_sites.pop()
    _test_site = test_sites.pop()

    train_validate_dataset = site_datasets["tvt"][_train_site]["dataset"]
    train_validate_size = len(site_datasets["tvt"][_train_site]["subject_ids"])

    test_dataset = site_datasets["tvt"][_test_site]["dataset"]
    test_size = len(site_datasets["tvt"][_test_site]["subject_ids"])
    test_subjects = site_datasets["tvt"][_test_site]["subject_ids"]

    report_dataset = site_datasets["report"][_test_site]["dataset"]
    report_size = len(site_datasets["report"][_test_site]["subject_ids"])
    report_subjects = site_datasets["report"][_test_site]["subject_ids"]

    for _train_site in train_sites:
        train_validate_dataset = train_validate_dataset.concatenate(
            site_datasets["tvt"][_train_site]["dataset"], name="concatenate_train_sites"
        )

        train_validate_size += len(site_datasets["tvt"][_train_site]["subject_ids"])

    for _test_site in test_sites:
        test_dataset = test_dataset.concatenate(
            site_datasets["tvt"][_test_site]["dataset"], name="concatenate_test_sites"
        )
        report_dataset = report_dataset.concatenate(
            site_datasets["report"][_test_site]["dataset"],
            name="concatenate_report_sites",
        )

        test_size += len(site_datasets["tvt"][_test_site]["subject_ids"])
        report_size += len(site_datasets["report"][_test_site]["subject_ids"])

        test_subjects.extend(site_datasets["tvt"][_test_site]["subject_ids"])
        report_subjects.extend(site_datasets["report"][_test_site]["subject_ids"])

    train_validate_dataset = train_validate_dataset.shuffle(
        buffer_size=train_validate_size, seed=seed
    )

    train_size = int(train_fraction * train_validate_size)
    train_dataset = train_validate_dataset.take(train_size)
    validate_dataset = train_validate_dataset.skip(train_size)

    train_dataset = train_dataset.batch(batch_size=batch_size)
    validate_dataset = validate_dataset.batch(batch_size=batch_size)
    test_dataset = test_dataset.batch(batch_size=batch_size)
    report_dataset = report_dataset.batch(batch_size=batch_size)

    train_dataset = train_dataset.repeat(n_epochs)

    return {
        "datasets": {
            "train": train_dataset,
            "validate": validate_dataset,
            "test": test_dataset,
            "report": report_dataset,
        },
        "subjects": {
            "test": test_subjects,
            "report": report_subjects,
        },
        "sizes": {
            "train": train_size,
            "validate": train_validate_size - train_size,
            "test": test_size,
            "report": report_size,
        },
    }


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
    GCS_DATASET_DIR = op.join(GCS_DATA_PATH, "tfrecs", dataset_name, "all-data")

    if use_tpu:
        device_dataset_dir = GCS_DATASET_DIR
    else:
        LOCAL_DATASET_DIR = op.join(".", "tfrecs", dataset_name, "all-data")
        os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
        fs.get(GCS_DATASET_DIR, LOCAL_DATASET_DIR, recursive=True)
        device_dataset_dir = LOCAL_DATASET_DIR

    n_classes = 1
    batch_size = 16
    volume_shape = (128, 128, 128, n_channels)
    block_shape = (128, 128, 128, n_channels)
    num_parallel_calls = 8

    site_datasets = get_site_datasets(
        device_dataset_dir=device_dataset_dir,
        n_classes=n_classes,
        batch_size=batch_size,
        volume_shape=volume_shape,
        num_parallel_calls=num_parallel_calls,
    )

    datasets = split_datasets(
        site_datasets=site_datasets,
        train_sites_in=train_sites,
        test_sites_in=test_sites,
        batch_size=batch_size,
        n_epochs=n_epochs,
        seed=dataset_seed,
    )

    dataset_train = datasets["datasets"]["train"]
    dataset_validate = datasets["datasets"]["validate"]
    dataset_test = datasets["datasets"]["test"]
    dataset_report = datasets["datasets"]["report"]
    dataset_sizes = datasets["sizes"]

    if compute_volume_numbers:
        n_train_volumes = sum(len(y) for _, y in dataset_train.as_numpy_iterator())
        n_validate_volumes = sum(
            len(y) for _, y in dataset_validate.as_numpy_iterator()
        )
        n_test_volumes = sum(len(y) for _, y in dataset_test.as_numpy_iterator())
    else:
        n_train_volumes = dataset_sizes["train"] // batch_size * batch_size
        n_validate_volumes = dataset_sizes["validate"] // batch_size * batch_size
        n_test_volumes = dataset_sizes["test"] // batch_size * batch_size
        n_report_volumes = dataset_sizes["report"] // batch_size * batch_size

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
            dataset_train = dataset_train.map(
                label_y, num_parallel_calls=num_parallel_calls
            )
            dataset_validate = dataset_validate.map(
                label_y, num_parallel_calls=num_parallel_calls
            )
            dataset_test = dataset_test.map(
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
        dataset_train,
        batch_size=batch_size,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=dataset_validate,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    model.save(SAVED_MODEL_DIR)
    model.evaluate(dataset_test, steps=test_steps)

    fs.put(LOCAL_MODEL_CHECKPOINT_DIR, MODEL_CHECKPOINT_DIR, recursive=True)
    fs.put(LOCAL_CSV_LOGGER_FILEPATH, CSV_LOGGER_FILEPATH)

    print("Predicting the report set.")
    y_hat_test = model.predict(dataset_test, steps=test_steps)
    y_test = np.concatenate([y.numpy() for _, y in dataset_test])
    df_report = pd.DataFrame(
        dict(
            subject_id=datasets["subjects"]["test"],
            y_true=np.squeeze(y_test),
            y_prob=np.squeeze(y_hat_test),
        )
    )
    df_report.to_csv(op.join(LOCAL_PREDICTION_OUTPUT_DIR, "test_set.csv"))

    print("Predicting the report set.")
    y_hat_report = model.predict(dataset_report, steps=report_steps)
    y_report = np.concatenate([y.numpy() for _, y in dataset_report])
    df_report = pd.DataFrame(
        dict(
            subject_id=datasets["subjects"]["report"],
            y_true=np.squeeze(y_report),
            y_prob=np.squeeze(y_hat_report),
        )
    )
    df_report.to_csv(op.join(LOCAL_PREDICTION_OUTPUT_DIR, "report_set.csv"))
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