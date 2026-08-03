"""
EyeQ dataset loader for Image Quality Assessment (pipeline Step 1).

Reads datasets/EyeQ/raw/{train,test}/labels.csv (columns: image, quality,
DR_grade) and the corresponding images/ folders, and builds tf.data.Dataset
pipelines directly from the original RGB images -- no CLAHE / Ben Graham /
green-channel or other preprocessing is applied here, since the IQA model is
required to learn from the unmodified fundus images. datasets/EyeQ/raw is
read-only from this module's perspective; nothing here writes into it.

Quality label convention (EyeQ dataset): 0 = Good, 1 = Usable, 2 = Reject.
"""

import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import EYEQ_RAW_DIR

QUALITY_CLASSES = ["Good", "Usable", "Reject"]
NUM_CLASSES = len(QUALITY_CLASSES)


def _read_labels(raw_dir, split):
    """Load a split's labels.csv and resolve each image's on-disk path.
    Rows whose image file is missing are dropped (and reported) rather than
    failing the whole load."""
    csv_path = os.path.join(raw_dir, split, "labels.csv")
    images_dir = os.path.join(raw_dir, split, "images")
    df = pd.read_csv(csv_path)
    df["path"] = df["image"].apply(lambda name: os.path.join(images_dir, name))

    exists = df["path"].apply(os.path.exists)
    if not exists.all():
        print(f"[image_quality_dataset] {(~exists).sum()} of {len(df)} images listed in "
              f"{csv_path} were not found under {images_dir} and will be skipped.")
        df = df[exists].reset_index(drop=True)
    return df


def _decode_image(path, image_size):
    raw = tf.io.read_file(path)
    image = tf.io.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, image_size)
    return tf.cast(image, tf.float32)


def _make_dataset(df, image_size, batch_size, shuffle, seed=42):
    paths = df["path"].values
    labels = tf.keras.utils.to_categorical(df["quality"].values, num_classes=NUM_CLASSES)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda path, label: (_decode_image(path, image_size), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_eyeq_datasets(raw_dir=EYEQ_RAW_DIR, image_size=(224, 224), batch_size=16,
                        val_split=0.15, seed=42):
    """
    Stratified split of datasets/EyeQ/raw/train/ into training and
    validation tf.data.Dataset pipelines (each yielding RGB float32 images
    in [0, 255] and one-hot quality labels), plus a class_weight dict
    computed from the training split to counter EyeQ's quality-class
    imbalance (Good is the large majority).

    raw/test/ is intentionally left untouched here -- see
    `load_eyeq_test_split` for the held-out evaluation split.
    """
    train_df = _read_labels(raw_dir, "train")
    train_split_df, val_split_df = train_test_split(
        train_df, test_size=val_split, stratify=train_df["quality"], random_state=seed,
    )

    train_ds = _make_dataset(train_split_df, image_size, batch_size, shuffle=True, seed=seed)
    val_ds = _make_dataset(val_split_df, image_size, batch_size, shuffle=False)

    class_counts = train_split_df["quality"].value_counts().sort_index()
    total = float(class_counts.sum())
    class_weights = {
        int(cls): total / (NUM_CLASSES * count)
        for cls, count in class_counts.items()
    }

    return train_ds, val_ds, class_weights


def load_eyeq_test_split(raw_dir=EYEQ_RAW_DIR, image_size=(224, 224), batch_size=16):
    """
    The held-out raw/test/ split: an unshuffled, image-only tf.data.Dataset
    ready for model.predict(), plus the aligned integer labels and
    filenames (same row order) for evaluation/inference scripts.
    """
    test_df = _read_labels(raw_dir, "test")
    image_ds = (
        tf.data.Dataset.from_tensor_slices(test_df["path"].values)
        .map(lambda path: _decode_image(path, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    labels = test_df["quality"].values.astype(int)
    filenames = test_df["image"].values
    return image_ds, labels, filenames
