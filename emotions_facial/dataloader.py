import pathlib
import os
import tensorflow as tf


def _prepare_label_lookup(directory):
    file_paths = tf.data.Dataset.list_files(
        str(pathlib.Path(directory) / "*/*"), shuffle=False
    )
    labels = file_paths.map(lambda x: tf.strings.split(x, os.path.sep)[-2])
    label_lookup = tf.keras.layers.StringLookup(num_oov_indices=0)
    label_lookup.adapt(labels)
    return label_lookup


def _encode_label(label, label_lookup):
    label_id = label_lookup(label)
    depth = tf.cast(label_lookup.vocabulary_size(), tf.int32)
    return tf.one_hot(label_id, depth=depth)


def _process_path(file_path, label_lookup):
    # Read the image from the file
    image = tf.io.read_file(file_path)
    # Decode the image based on file extension
    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(image, channels=3),
        lambda: tf.image.decode_png(image, channels=3),
    )
    # Resize the image to (48, 48)
    image = tf.image.resize(image, [48, 48])
    # Normalize the image to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # Extract label from file path
    parts = tf.strings.split(file_path, "/")
    label = parts[-2]
    # Look up the label index
    label = _encode_label(label, label_lookup)
    return image, label


def load(train_dir, test_dir):
    label_lookup = _prepare_label_lookup(train_dir)
    # Load the training dataset
    train_ds = tf.data.Dataset.list_files(
        str(pathlib.Path(train_dir) / "*/*"), shuffle=True
    )
    train_ds = train_ds.map(
        lambda x: _process_path(x, label_lookup),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    # Load the test dataset for jpg and png files
    test_ds = tf.data.Dataset.list_files(
        str(pathlib.Path(test_dir) / "*/*"), shuffle=False
    )
    test_ds = test_ds.map(
        lambda x: _process_path(x, label_lookup),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return test_ds, train_ds
