import pathlib
import tensorflow as tf


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
    label = tf.strings.as_string(label)
    label = label_lookup[label.numpy().decode("utf-8")]
    return image, label


def load(train_dir, test_dir):
    class_names = sorted(
        [item.name for item in pathlib.Path(train_dir).glob("*/") if item.is_dir()]
    )
    label_lookup = {name: idx for idx, name in enumerate(class_names)}

    # Load the training dataset
    train_jpg_ds = tf.data.Dataset.list_files(
        str(pathlib.Path(train_dir) / "*/*.jpg"), shuffle=True
    )
    train_png_ds = tf.data.Dataset.list_files(
        str(pathlib.Path(train_dir) / "*/*.png"), shuffle=True
    )

    train_ds = train_jpg_ds.concatenate(train_png_ds)
    train_ds = train_ds.map(
        lambda x: _process_path(x, label_lookup),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # Load the test dataset for jpg and png files
    test_jpg_ds = tf.data.Dataset.list_files(
        str(pathlib.Path(test_dir) / "*/*.jpg"), shuffle=False
    )
    test_png_ds = tf.data.Dataset.list_files(
        str(pathlib.Path(test_dir) / "*/*.png"), shuffle=False
    )

    test_ds = test_jpg_ds.concatenate(test_png_ds)
    test_ds = test_ds.map(
        lambda x: _process_path(x, label_lookup),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    return test_ds, train_ds
