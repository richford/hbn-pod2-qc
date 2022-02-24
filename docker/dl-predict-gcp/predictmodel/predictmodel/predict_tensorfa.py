import argparse
import gcsfs
import nobrainer
import numpy as np
import os
import os.path as op
import pandas as pd
import tensorflow as tf


def label_y(x, y):
    return x, tf.cast(y >= 0.5, dtype=tf.int32)


def main(
    gcs_bucket,
    job_name,
    n_channels=3,
    dataset_name="b0-tensorfa-dwiqc",
    dataset_seed=0,
    model_loss="binary_crossentropy",
    compute_volume_numbers=False,
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
    GCS_BASE_PATH = f"gs://{gcs_bucket}/{job_name}/seed_{dataset_seed}"
    GCS_SAVED_MODEL_DIR = op.join(GCS_BASE_PATH, "saved_model")
    GCS_PREDICTION_OUTPUT_DIR = op.join(GCS_BASE_PATH, "output")

    fs = gcsfs.GCSFileSystem()
    fs.touch(op.join(GCS_SAVED_MODEL_DIR, "assets", ".gcskeep"))

    LOCAL_SAVED_MODEL_DIR = "saved_model"
    LOCAL_PREDICTION_OUTPUT_DIR = "output"
    os.makedirs(LOCAL_SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(LOCAL_PREDICTION_OUTPUT_DIR, exist_ok=True)

    fs.get(GCS_SAVED_MODEL_DIR, LOCAL_SAVED_MODEL_DIR, recursive=True)

    # Specify the datasets on GCP storage
    GCS_DATA_PATH = f"gs://{gcs_bucket}"
    GCS_DATASET_DIR = op.join(
        GCS_DATA_PATH, "tfrecs", dataset_name, f"seed_{dataset_seed}"
    )
    GCS_ALLDATA_DIR = op.join(GCS_DATA_PATH, "tfrecs", dataset_name, "all-data")

    if use_tpu:
        device_dataset_dir = GCS_DATASET_DIR
        device_alldata_dir = GCS_ALLDATA_DIR
    else:
        LOCAL_DATASET_DIR = op.join(".", "tfrecs", dataset_name, f"seed_{dataset_seed}")
        LOCAL_ALLDATA_DIR = op.join(".", "tfrecs", dataset_name, "all-data")

        os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
        os.makedirs(LOCAL_ALLDATA_DIR, exist_ok=True)

        fs.get(GCS_DATASET_DIR, LOCAL_DATASET_DIR, recursive=True)
        fs.get(GCS_ALLDATA_DIR, LOCAL_ALLDATA_DIR, recursive=True)

        device_dataset_dir = LOCAL_DATASET_DIR
        device_alldata_dir = LOCAL_ALLDATA_DIR

    n_classes = 1
    batch_size = 1
    volume_shape = (128, 128, 128, n_channels)
    block_shape = (128, 128, 128, n_channels)
    num_parallel_calls = 1

    dataset_report = nobrainer.dataset.get_dataset(
        file_pattern=op.join(device_dataset_dir, "data-report_shard*.tfrec"),
        n_classes=n_classes,
        batch_size=batch_size,
        volume_shape=volume_shape,
        scalar_label=True,
        # block_shape=block_shape,
        augment=False,
        n_epochs=1,
        num_parallel_calls=1,
    )

    dataset_all = nobrainer.dataset.get_dataset(
        file_pattern=op.join(device_alldata_dir, "data-all_shard-*.tfrec"),
        n_classes=n_classes,
        batch_size=batch_size,
        volume_shape=volume_shape,
        scalar_label=True,
        # block_shape=block_shape,
        augment=False,
        n_epochs=1,
        num_parallel_calls=1,
    )

    if compute_volume_numbers:
        n_report_volumes = sum(len(y) for _, y in dataset_report.as_numpy_iterator())
        n_all_volumes = sum(len(y) for _, y in dataset_all.as_numpy_iterator())
    else:
        n_report_volumes = 200 // batch_size * batch_size
        n_all_volumes = 2129 // batch_size * batch_size

    print("n_report_volumes: ", n_report_volumes)
    print("n_all_volumes: ", n_all_volumes)

    report_steps = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_report_volumes,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_size,
    )

    all_steps = nobrainer.dataset.get_steps_per_epoch(
        n_volumes=n_all_volumes,
        volume_shape=volume_shape,
        block_shape=block_shape,
        batch_size=batch_size,
    )

    if use_tpu:
        compile_kwargs = {"steps_per_execution": report_steps // 2}
    else:
        compile_kwargs = {}

    y_all_float = np.concatenate([y.numpy() for _, y in dataset_all])

    with scope:
        model = tf.keras.models.load_model(LOCAL_SAVED_MODEL_DIR)

        if model_loss == "mean_squared_error":
            # Define binary accuracy and AUC metrics when y_true is a probability
            def binary_accuracy_with_prob(y_prob, y_pred):
                y_true = tf.cast(tf.round(y_prob), tf.int32)
                m = tf.keras.metrics.BinaryAccuracy()
                m.update_state(y_true, y_pred)
                return m.result().numpy()

            model.compile(
                loss=model.loss,
                optimizer=model.optimizer,
                metrics=[
                    model_loss,
                    binary_accuracy_with_prob,
                ],
                **compile_kwargs,
                # run_eagerly=True,
            )
        elif model_loss == "binary_crossentropy":
            dataset_report = dataset_report.map(
                label_y, num_parallel_calls=num_parallel_calls
            )
            dataset_all = dataset_all.map(
                label_y, num_parallel_calls=num_parallel_calls
            )

            model.compile(
                loss=model.loss,
                optimizer=model.optimizer,
                metrics=[
                    model_loss,
                    "accuracy",
                    "AUC",
                ],
                **compile_kwargs,
                # run_eagerly=True,
            )

    print(model.summary())

    print("Predicting the report set.")
    y_hat_report = model.predict(dataset_report, steps=report_steps)
    y_report = np.concatenate([y.numpy() for _, y in dataset_report])
    df_report = pd.DataFrame(
        dict(y_true=np.squeeze(y_report), y_prob=np.squeeze(y_hat_report))
    )
    df_report.to_csv(op.join(LOCAL_PREDICTION_OUTPUT_DIR, "report_set.csv"))

    print("Predicting the entire dataset.")
    y_hat_all = model.predict(dataset_all, steps=all_steps)
    y_all = np.concatenate([y.numpy() for _, y in dataset_all])
    df_all = pd.DataFrame(
        dict(
            y_true_float=np.squeeze(y_all_float),
            y_true_binarized=np.squeeze(y_all),
            y_prob=np.squeeze(y_hat_all),
        )
    )
    df_all.to_csv(op.join(LOCAL_PREDICTION_OUTPUT_DIR, "all_data.csv"))

    fs.put(LOCAL_PREDICTION_OUTPUT_DIR, GCS_PREDICTION_OUTPUT_DIR, recursive=True)


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
        "--n_channels",
        type=int,
        help="The number of channels in the data.",
        default=3,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The name of the dataset in the tfrecs folder of the GCS bucket.",
        default="b0-colorfa-rgb",
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
    parser.set_defaults(compute_volume_numbers=False)

    args = parser.parse_args()

    main(
        gcs_bucket=args.gcs_bucket,
        job_name=args.job_name,
        n_channels=args.n_channels,
        dataset_name=args.dataset_name,
        dataset_seed=args.dataset_seed,
        model_loss=args.model_loss,
        compute_volume_numbers=args.compute_volume_numbers,
    )
