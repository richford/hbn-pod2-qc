import argparse
import gcsfs
import nobrainer
import os
import os.path as op
import tensorflow as tf


MODEL_PATH = op.join(op.dirname(__file__), "brain-extraction-unet-128iso-model.h5")


def unet_slice_model(
    model_name,
    model_path=MODEL_PATH,
    layer_prefix="",
    layer_postfix="",
    input_shape=(128, 128, 128, 1),
    l2_reg=0.001,
    trainable=False,
):
    model = tf.keras.models.load_model(model_path, compile=False)
    model._name = model_name
    if trainable:
        for layer in model.layers:
            layer.kernel_regularizer = tf.keras.regularizers.l2(l2_reg)
    else:
        model.trainable = False

    def new_name(name):
        return "".join([layer_prefix, name, layer_postfix])

    for layer in model.layers:
        layer._name = new_name(layer.name)

    layer_names = [layer.name for layer in model.layers]
    use_layers = layer_names[1 : layer_names.index(new_name("activation_6")) + 1]

    _input = tf.keras.Input(shape=input_shape, name=new_name("input"))
    l = _input
    for layer_name in use_layers:
        l = model.get_layer(layer_name)(l)

    return tf.keras.Model(_input, l)


class SliceChannel(tf.keras.layers.Layer):
    def __init__(self, name, channel):
        super(SliceChannel, self).__init__(trainable=False, name=name)
        self.channel = channel
    
    def call(self, inputs):
        return inputs[:, :, :, :, self.channel]
    
    def get_config(self):
        return {"name": self.name, "channel": self.channel}


def create_unet_model_4channels(unet_model_path=MODEL_PATH):
    _input = tf.keras.layers.Input(shape=(128, 128, 128, 4))

    channel0 = SliceChannel(name=f"channel_0", channel=0)(_input)
    channel1 = SliceChannel(name=f"channel_1", channel=1)(_input)
    channel2 = SliceChannel(name=f"channel_2", channel=2)(_input)
    channel3 = SliceChannel(name=f"channel_3", channel=3)(_input)

    unet = unet_slice_model(model_path=unet_model_path, model_name="unet")

    unet_models = [unet(_channel) for _channel in [
        channel0, channel1, channel2, channel3
    ]]

    x = tf.keras.layers.concatenate(unet_models)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=1)(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
    model = tf.keras.Model(_input, outputs, name="unet_4channel")
    return model


def create_unet_model_3channels(unet_model_path=MODEL_PATH):
    _input = tf.keras.layers.Input(shape=(128, 128, 128, 3))

    channel0 = SliceChannel(name=f"channel_0", channel=0)(_input)
    channel1 = SliceChannel(name=f"channel_1", channel=1)(_input)
    channel2 = SliceChannel(name=f"channel_2", channel=2)(_input)

    unet = unet_slice_model(model_path=unet_model_path, model_name="unet")

    unet_models = [unet(_channel) for _channel in [
        channel0, channel1, channel2
    ]]

    x = tf.keras.layers.concatenate(unet_models)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=1)(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
    model = tf.keras.Model(_input, outputs, name="unet_3channel")
    return model


def create_unet_model(unet_model_path=MODEL_PATH, n_channels=3):
    _input = tf.keras.layers.Input(shape=(128, 128, 128, n_channels))
    unet = unet_slice_model(model_path=unet_model_path, model_name="unet")

    unet_models = []
    for channel in range(n_channels):
        _channel = SliceChannel(name=f"channel_{channel}", channel=channel)(_input)
        unet_models.append(unet(_channel))

    x = tf.keras.layers.concatenate(unet_models)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=1)(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)
    model = tf.keras.Model(_input, outputs, name=f"unet_{n_channels}channel")
    return model


def build_multi_input_model(
    width=128, height=128, depth=128, n_image_channels=3, n_qc_metrics=30
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

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
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
    n_epochs=5,
    n_channels=3,
    dataset_name="b0-colorfa-rgb",
    dataset_seed=0,
    model_loss="binary_crossentropy",
    compute_volume_numbers=False,
):
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
    GCS_BASE_PATH = f"gs://{gcs_bucket}/{job_name}"
    TENSORBOARD_LOGS_DIR = op.join(GCS_BASE_PATH, "logs")
    MODEL_CHECKPOINT_DIR = op.join(GCS_BASE_PATH, "checkpoints")
    CSV_LOGGER_FILEPATH = op.join(
        GCS_BASE_PATH, "checkpoints", f"training_{dataset_seed}.csv"
    )
    SAVED_MODEL_DIR = op.join(GCS_BASE_PATH, "saved_model")

    fs = gcsfs.GCSFileSystem()
    fs.touch(CSV_LOGGER_FILEPATH)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("saved_model", exist_ok=True)

    LOCAL_MODEL_CHECKPOINT_DIR = "checkpoints"
    LOCAL_MODEL_CHECKPOINT_FILEPATH = op.join(
        LOCAL_MODEL_CHECKPOINT_DIR, "model_{epoch:03d}-{val_loss:.2f}.h5"
    )
    LOCAL_CSV_LOGGER_FILEPATH = op.join("checkpoints", f"training_{dataset_seed}.csv")

    # Specify the datasets on GCP storage
    GCS_DATA_PATH = f"gs://{gcs_bucket}"
    GCS_DATASET_DIR = op.join(
        GCS_DATA_PATH, "tfrecs", dataset_name, f"seed_{dataset_seed}"
    )

    if use_tpu:
        device_dataset_dir = GCS_DATASET_DIR
    else:
        LOCAL_DATASET_DIR = op.join(".", "tfrecs", dataset_name, f"seed_{dataset_seed}")
        os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)
        fs.get(GCS_DATASET_DIR, LOCAL_DATASET_DIR, recursive=True)
        device_dataset_dir = LOCAL_DATASET_DIR

    n_classes = 1
    batch_size = 16
    volume_shape = (128, 128, 128, n_channels)
    block_shape = (128, 128, 128, n_channels)
    num_parallel_calls = 2

    dataset_train = nobrainer.dataset.get_dataset(
        file_pattern=op.join(device_dataset_dir, "data-train_shard*.tfrec"),
        n_classes=n_classes,
        batch_size=batch_size,
        volume_shape=volume_shape,
        scalar_label=True,
        # block_shape=block_shape,
        augment=True,
        n_epochs=None,
        num_parallel_calls=num_parallel_calls,
    )

    dataset_validate = nobrainer.dataset.get_dataset(
        file_pattern=op.join(device_dataset_dir, "data-validate_shard-*.tfrec"),
        n_classes=n_classes,
        batch_size=batch_size,
        volume_shape=volume_shape,
        scalar_label=True,
        # block_shape=block_shape,
        augment=True,
        n_epochs=1,
        num_parallel_calls=num_parallel_calls,
    )

    dataset_test = nobrainer.dataset.get_dataset(
        file_pattern=op.join(device_dataset_dir, "data-test_shard-*.tfrec"),
        n_classes=n_classes,
        batch_size=batch_size,
        volume_shape=volume_shape,
        scalar_label=True,
        # block_shape=block_shape,
        augment=True,
        n_epochs=1,
        num_parallel_calls=num_parallel_calls,
    )

    if compute_volume_numbers:
        n_train_volumes = sum(len(y) for _, y in dataset_train.as_numpy_iterator())
        n_validate_volumes = sum(
            len(y) for _, y in dataset_validate.as_numpy_iterator()
        )
        n_test_volumes = sum(len(y) for _, y in dataset_test.as_numpy_iterator())
    else:
        n_train_volumes = 1156 // batch_size * batch_size
        n_validate_volumes = 144 // batch_size * batch_size
        n_test_volumes = 146 // batch_size * batch_size

    print("n_train_volumes: ", n_train_volumes)
    print("n_validate_volumes: ", n_validate_volumes)
    print("n_test_volumes: ", n_test_volumes)

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

    if use_tpu:
        compile_kwargs = {"steps_per_execution": steps_per_epoch // 2}
    else:
        compile_kwargs = {}

    with scope:
        # Set learning rate schedule
        initial_learning_rate = 0.0001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        if model_loss == "mean_squared_error":
            # Define binary accuracy and AUC metrics when y_true is a probability
            def binary_accuracy_with_prob(y_prob, y_pred):
                y_true = tf.cast(tf.round(y_prob), tf.int32)
                m = tf.keras.metrics.BinaryAccuracy()
                m.update_state(y_true, y_pred)
                return m.result().numpy()

            if n_channels == 3:
                model = create_unet_model_3channels()
            elif n_channels == 4:
                model = create_unet_model_4channels()
            else:
                model = create_unet_model(n_channels=n_channels)

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

            if n_channels == 3:
                model = create_unet_model_3channels()
            elif n_channels == 4:
                model = create_unet_model_4channels()
            else:
                model = create_unet_model(n_channels=n_channels)

            model.compile(
                loss=model_loss,
                optimizer=optimizer,
                metrics=[
                    model_loss,
                    "accuracy",
                    tf.keras.metrics.AUC(),
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

    # Configure Tensorboard logs
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOGS_DIR),
        best_model,
        csv_logger,
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0.001,
            patience=15,
        ),
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
        "--n_epochs",
        type=int,
        help="The number of training epochs.",
        default=5,
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
        n_epochs=args.n_epochs,
        n_channels=args.n_channels,
        dataset_name=args.dataset_name,
        dataset_seed=args.dataset_seed,
        model_loss=args.model_loss,
        compute_volume_numbers=args.compute_volume_numbers,
    )
