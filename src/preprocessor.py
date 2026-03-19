from typing import Literal

import tensorflow

MEAN = [0.485, 0.456, 0.406]
STANDARD_DEVIATION = [0.229, 0.224, 0.225]
MEAN_TF = tensorflow.reshape(
    tensorflow.constant(MEAN, dtype=tensorflow.float32), [1, 1, 3]
)
STD_TF = tensorflow.reshape(
    tensorflow.constant(STANDARD_DEVIATION, dtype=tensorflow.float32), [1, 1, 3]
)


def normalize_images(
    ds: tensorflow.data.Dataset, normalization: Literal["none", "imagenet"] = "none"
) -> tensorflow.data.Dataset:
    """Normalizes images in a tensorflow.data.Dataset pipeline.

    This function converts input images to float32, scales them to the [0, 1]
    range, and optionally applies ImageNet normalization using predefined
    channel-wise mean and standard deviation.

    Args:
        ds: A `tensorflow.data.Dataset` yielding tuples of the form (image, label),
            where images are expected to be uint8 tensors.
        normalization: Type of normalization to apply:
            - "none": only rescales images to [0, 1]
            - "imagenet": applies ImageNet mean/std normalization

    Returns:
        A `tensorflow.data.Dataset` where:
            - images are float32 tensors, normalized according to the selected mode
            - labels are unchanged

    Notes:
        - ImageNet normalization uses mean = [0.485, 0.456, 0.406] and
          std = [0.229, 0.224, 0.225].
        - The transformation is applied lazily via `Dataset.map`.
        - Parallel mapping is enabled via `AUTOTUNE` for better performance.
    """

    def preprocess(x, y):
        x = tensorflow.cast(x, tensorflow.float32) / 255.0

        if normalization == "imagenet":
            x = (x - MEAN_TF) / STD_TF

        return x, y

    return ds.map(preprocess, num_parallel_calls=tensorflow.data.AUTOTUNE)


NUMBER_CLASSES = 100
NUMBER_SUPER_CLASSES = 20


def encode_labels(
    ds: tensorflow.data.Dataset,
    label_mode: Literal["fine", "coarse"] = "fine",
) -> tensorflow.data.Dataset:
    """Encodes labels into one-hot vectors for classification tasks.

    This function transforms integer class labels into one-hot encoded vectors,
    supporting both fine-grained (100 classes) and coarse (20 super-classes)
    label schemes.

    Args:
        ds: A `tensorflow.data.Dataset` yielding tuples of the form (image, label),
            where labels are integer tensors.
        label_mode: Label granularity to encode:
            - "fine": encodes into 100 classes (default, CIFAR-100 fine labels)
            - "coarse": encodes into 20 super-classes

    Returns:
        A `tensorflow.data.Dataset` where:
            - images are unchanged
            - labels are one-hot encoded tensors of shape (num_classes,)

    Notes:
        - Labels are reshaped to 1D using `tensorflow.reshape` to ensure compatibility.
        - One-hot encoding is required for techniques such as MixUp.
        - Parallel mapping is enabled via `AUTOTUNE`.
    """
    num_classes = NUMBER_SUPER_CLASSES if label_mode == "coarse" else NUMBER_CLASSES

    def encode(x, y):
        y = tensorflow.cast(y, tensorflow.int32)
        y = tensorflow.reshape(y, [-1])
        y = tensorflow.one_hot(y, num_classes)
        return x, y

    return ds.map(encode, num_parallel_calls=tensorflow.data.AUTOTUNE)


def get_augmentation_layer(
    rotation: float = 0.2,
    zoom: float = 0.25,
    shift: float = 0.15,
    flip: str = "horizontal",
    brightness: float = 0.2,
    erasing_factor: float = 0.2,
    erasing_fill_value: float = 0.0,
) -> tensorflow.keras.Sequential:
    """Creates a data augmentation pipeline.

    The augmentation pipeline applies geometric transformations first,
    followed by photometric transformations and finally random erasing.

    Args:
        rotation: Rotation factor for `RandomRotation`.
        zoom: Zoom factor for `RandomZoom`.
        shift: Translation factor for `RandomTranslation`.
        flip: Flip mode. One of:
            - "horizontal"
            - "vertical"
            - "horizontal_and_vertical"
        brightness: Brightness adjustment factor.
        erasing_factor: Fraction of the image to erase.
        erasing_fill_value: Value used to fill erased regions.

    Returns:
        A `tensorflow.keras.Sequential` model representing the augmentation pipeline.
    """
    return tensorflow.keras.Sequential(
        [
            # Geometric transformations
            tensorflow.keras.layers.RandomRotation(rotation),
            tensorflow.keras.layers.RandomZoom(zoom),
            tensorflow.keras.layers.RandomTranslation(
                height_factor=shift,
                width_factor=shift,
            ),
            # Flip
            tensorflow.keras.layers.RandomFlip(flip),
            # Photometric transformations
            tensorflow.keras.layers.RandomBrightness(factor=brightness),
            # Occlusion
            tensorflow.keras.layers.RandomErasing(
                factor=erasing_factor,
                fill_value=erasing_fill_value,
            ),
        ]
    )


def sample_beta_distribution(
    shape: tensorflow.Tensor, alpha: float = 0.2
) -> tensorflow.Tensor:
    """Samples values from a Beta distribution using Gamma distributions.

    The Beta distribution is constructed from two Gamma-distributed samples:
    Beta(a, b) = Gamma(a) / (Gamma(a) + Gamma(b)).

    Args:
        shape: Shape of the output tensor.
        alpha: Concentration parameter for both Gamma distributions (a = b = alpha).

    Returns:
        A TensorFlow tensor of shape `shape` containing samples from the
        Beta(alpha, alpha) distribution.
    """
    gamma1 = tensorflow.random.gamma(shape, alpha)
    gamma2 = tensorflow.random.gamma(shape, alpha)
    return gamma1 / (gamma1 + gamma2)


def mix_up_batches(
    batch_x: tensorflow.Tensor,
    batch_y: tensorflow.Tensor,
    alpha: float = 0.2,
) -> tuple[tensorflow.Tensor, tensorflow.Tensor]:
    """Applies MixUp augmentation to a batch of data.

    MixUp creates new training samples by linearly interpolating between
    pairs of examples within the same batch.

    Args:
        batch_x: Input batch of images, shape (batch_size, H, W, C).
        batch_y: Input batch of labels (one-hot encoded),
            shape (batch_size, num_classes).
        alpha: Concentration parameter for the Beta distribution used to
            sample interpolation weights.

    Returns:
        A tuple (mixed_x, mixed_y) where:
            - mixed_x: Mixed images with the same shape as `batch_x`
            - mixed_y: Mixed labels with the same shape as `batch_y`
    """
    batch_size = tensorflow.shape(batch_x)[0]

    lambda_ = sample_beta_distribution((batch_size, 1, 1, 1), alpha)
    lambda_y = tensorflow.reshape(lambda_, (batch_size, 1))

    indices = tensorflow.random.shuffle(tensorflow.range(batch_size))

    x2 = tensorflow.gather(batch_x, indices)
    y2 = tensorflow.gather(batch_y, indices)

    mixed_x = batch_x * lambda_ + x2 * (1 - lambda_)
    mixed_y = batch_y * lambda_y + y2 * (1 - lambda_y)

    return mixed_x, mixed_y


def mix_up_images(
    ds: tensorflow.data.Dataset,
    batch_size: int = 32,
) -> tensorflow.data.Dataset:
    """Applies MixUp augmentation to a tensorflow.data.Dataset.

    This function shuffles, batches, and applies MixUp augmentation to
    the dataset. MixUp is applied at the batch level by interpolating
    between shuffled samples.

    Args:
        ds: A `tensorflow.data.Dataset` yielding (image, label) pairs.
        batch_size: Number of samples per batch.

    Returns:
        A `tensorflow.data.Dataset` where:
            - inputs are mixed images
            - labels are linearly interpolated
    """
    ds = ds.shuffle(10000).batch(batch_size)

    def apply_mixup(x, y):
        return mix_up_batches(x, y)

    return ds.map(apply_mixup, num_parallel_calls=tensorflow.data.AUTOTUNE)


if __name__ == "__main__":
    label_mode = "fine"

    (X_train_fine, y_train_fine), (X_test_fine, y_test_fine) = (
        tensorflow.keras.datasets.cifar100.load_data(label_mode=label_mode)
    )

    train_ds_fine = tensorflow.data.Dataset.from_tensor_slices(
        (X_train_fine, y_train_fine)
    )
    test_ds_fine = tensorflow.data.Dataset.from_tensor_slices(
        (X_test_fine, y_test_fine)
    )

    train_ds_fine_normalized = normalize_images(train_ds_fine, normalization="none")
    test_ds_fine_normalized = normalize_images(test_ds_fine, normalization="none")

    train_ds_fine_encoded = encode_labels(
        train_ds_fine_normalized, label_mode=label_mode
    )
    test_ds_fine_encoded = encode_labels(test_ds_fine_normalized, label_mode=label_mode)

    augmentation_layer = get_augmentation_layer()

    train_ds_mix_up = encode_labels(train_ds_fine_encoded)
