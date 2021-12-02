import argparse
import gc
import gcsfs
import nibabel as nib
import nilearn
import nobrainer
import numpy as np
import os
import os.path as op
import pandas as pd
import tensorflow as tf


def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images


def compute_gradients(model, images, target_class):
    with tf.GradientTape() as tape:
        tape.watch(images)
        raw_probs = model(images)
        probs = (1 - raw_probs) * (1 - target_class) + raw_probs * target_class

    gradients = tape.gradient(probs, images)
    return gradients


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    return tf.math.reduce_mean(grads, axis=0)


@tf.function
def integrated_gradients(
    model, baseline, image, target_class, m_steps=50, batch_size=32
):
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)

    # Initialize TensorArray outside loop to collect gradients.
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps + 1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_images(
            baseline=baseline, image=image, alphas=alpha_batch
        )

        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(
            model=model,
            images=interpolated_path_input_batch,
            target_class=target_class,
        )

        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    return (image - baseline) * avg_gradients


def main(
    gcs_bucket,
    n_channels=5,
    dataset_name="b0-tensorfa-dwiqc",
    model_dir="b0_tensorfa_dwiqc",
    dataset_seed=8,
    target_class=1,
    confusion_class="true_pos",
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

    # Setting location were training logs and checkpoints will be stored
    GCS_BASE_PATH = f"gs://{gcs_bucket}/{model_dir}/seed_{dataset_seed}"
    GCS_SAVED_MODEL_DIR = op.join(GCS_BASE_PATH, "saved_model")
    GCS_OUTPUT_DIR = op.join(GCS_BASE_PATH, "integrated_gradients")

    fs = gcsfs.GCSFileSystem()

    LOCAL_SAVED_MODEL_DIR = "saved_model"
    LOCAL_OUTPUT_DIR = "output"
    os.makedirs(LOCAL_SAVED_MODEL_DIR, exist_ok=True)
    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    fs.get(GCS_SAVED_MODEL_DIR, LOCAL_SAVED_MODEL_DIR, recursive=True)

    # Specify the datasets on GCP storage
    GCS_DATA_PATH = f"gs://{gcs_bucket}"
    GCS_ALLDATA_DIR = op.join(GCS_DATA_PATH, "tfrecs", dataset_name, "all-data")

    if use_tpu:
        device_alldata_dir = GCS_ALLDATA_DIR
    else:
        LOCAL_ALLDATA_DIR = op.join(".", "tfrecs", dataset_name, "all-data")
        os.makedirs(LOCAL_ALLDATA_DIR, exist_ok=True)
        fs.get(GCS_ALLDATA_DIR, LOCAL_ALLDATA_DIR, recursive=True)
        device_alldata_dir = LOCAL_ALLDATA_DIR

    volume_shape = (128, 128, 128, n_channels)
    element_spec = (
        tf.TensorSpec(shape=(), dtype=tf.int64, name=None),
        (
            tf.TensorSpec(shape=(1, 128, 128, 128, 5), dtype=tf.float32, name=None),
            tf.TensorSpec(shape=(1,), dtype=tf.float32, name=None),
        ),
    )

    dataset = tf.data.experimental.load(
        op.join(device_alldata_dir, confusion_class),
        element_spec=element_spec,
    )
    volumes = [tf.squeeze(tensor[0]) for _, tensor in dataset]
    baseline = tf.zeros(shape=volume_shape, dtype=tf.float32)

    print("Computing integrated gradients")

    with scope:
        model = tf.keras.models.load_model(LOCAL_SAVED_MODEL_DIR)

        ig_attributions = [
            integrated_gradients(
                model=model,
                baseline=baseline,
                image=volume,
                target_class=target_class,
                m_steps=128,
                batch_size=1,
            )
            for volume in volumes
        ]

    if target_class == 1:
        postfix = "attribution_pass"
    else:
        postfix = "attribution_fail"

    ig_dataset = tf.data.Dataset.from_tensor_slices(tf.stack(ig_attributions))
    tf.data.experimental.save(
        ig_dataset,
        op.join(LOCAL_OUTPUT_DIR, f"ig_{confusion_class}_{postfix}"),
    )

    affine = np.diag([1, 1, 1, 1])
    volume_niftis = [
        {
            "b0": nib.Nifti1Image(volume[:, :, :, 3].numpy(), affine),
            "color_fa": nib.Nifti1Image(volume[:, :, :, :3].numpy(), affine),
        }
        for volume in volumes
    ]
    ig_niftis = [
        {
            "b0": nib.Nifti1Image(attribution[:, :, :, 3].numpy(), affine),
            "color_fa": nib.Nifti1Image(attribution[:, :, :, :3].numpy(), affine),
            "sum": nib.Nifti1Image(
                tf.math.reduce_sum(attribution[:, :, :, :4], axis=-1).numpy(), affine
            ),
        }
        for attribution in ig_attributions
    ]

    for idx, (volume_nifti, ig_nifti) in enumerate(zip(volume_niftis, ig_niftis)):
        for key, value in volume_nifti.items():
            nib.save(
                value,
                op.join(LOCAL_OUTPUT_DIR, f"{confusion_class}_{key}_{idx}.nii.gz"),
            )

        for key, value in ig_nifti.items():
            nib.save(
                value,
                op.join(
                    LOCAL_OUTPUT_DIR, f"{confusion_class}_{postfix}_{key}_{idx}.nii.gz"
                ),
            )

    fs.put(LOCAL_OUTPUT_DIR, GCS_OUTPUT_DIR, recursive=True)


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
        "--model_dir",
        type=str,
        help="The name of the GCS directory in which the tensorflow model is saved.",
        default="b0_tensorfa_dwiqc",
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        help="The seed for the dataset",
        default=8,
    )
    parser.add_argument(
        "--target_class",
        type=int,
        help="The target class for the integrated gradients.",
        default=1,
    )
    parser.add_argument(
        "--confusion_class",
        type=str,
        help="The confusion class for which to compute integrated gradients",
        default="true_pos",
    )

    args = parser.parse_args()

    main(
        gcs_bucket=args.gcs_bucket,
        n_channels=args.n_channels,
        dataset_name=args.dataset_name,
        model_dir=args.model_dir,
        dataset_seed=args.dataset_seed,
        target_class=args.target_class,
        confusion_class=args.confusion_class,
    )
