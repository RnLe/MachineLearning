import tensorflow as tf


def _normalize_image(img):
    '''
    Normalize the image to the range [0,1]
    '''
    img = tf.cast(img, tf.float32) / 255.0
    return img

def _one_hot_encode(image, label, num_classes):
    '''
    One-hot encode the labels
    '''
    label = tf.one_hot(label, depth=num_classes)
    return image, label

def _augment(img, label, OneChannelOnly):
    '''
    Augment the image with random flips, brightness, contrast, hue and saturation.
    '''
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.5)
    img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
    # Some augmentations are only applicable to RGB images
    if not OneChannelOnly:
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    return img, label

def _augment_multiple(img, label, num_augmentations, OneChannelOnly):
    '''
    Augment the image multiple times.
    '''
    augmented_images = []
    augmented_labels = []
    for _ in range(num_augmentations):
        augmented_img, augmented_label = _augment(img, label, OneChannelOnly)
        augmented_images.append(augmented_img)
        augmented_labels.append(augmented_label)
    return tf.data.Dataset.from_tensor_slices((augmented_images, augmented_labels))

def _loadFromDirectory(train_dir, test_dir, OneChannelOnly, image_size, batch_size):
    '''
    Load the data from the directory.
    
    Args
    ------------
    - train_dir: The path to the training directory
    - test_dir: The path to the testing directory
    - OneChannelOnly: If True, the images will be loaded as grayscale images
    '''
    
    # Load the data with inferred classes (from subfolders)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',  # Integer-Labels
        image_size=image_size,
        batch_size=batch_size,
        color_mode='grayscale' if OneChannelOnly else 'rgb',
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='int',  # Integer-Labels
        image_size=image_size,
        batch_size=batch_size,
        color_mode='grayscale' if OneChannelOnly else 'rgb',
    )
    
    return test_ds, train_ds
    

def load(train_dir, test_dir, num_classes, batch_size=64, augmentations=None, image_size=(48, 48), OneChannelOnly=False):
    '''
    Load training and testing data from the given directories. Classes are inferred from the subfolders.
    
    Args
    ------------
    - train_dir: The path to the training directory
    - test_dir: The path to the testing directory
    - num_classes: The number of classes
    - batch_size: The batch size
    - augmentations: The number of augmentations to apply to the training data
    - image_size: The size of the images
    - OneChannelOnly: If True, the images will be loaded as grayscale images
    
    Returns
    ------------
    - test_ds: The testing dataset
    - train_ds: The training dataset
    '''
    test_ds, train_ds = _loadFromDirectory(train_dir, test_dir, OneChannelOnly, image_size, batch_size)
    
    # Normalize the images to [0,1]
    train_ds = train_ds.map(lambda img, label: (_normalize_image(img), label))
    test_ds = test_ds.map(lambda img, label: (_normalize_image(img), label))
    
    # One-hot encode the labels
    train_ds = train_ds.map(lambda img, label: _one_hot_encode(img, label, num_classes))
    test_ds = test_ds.map(lambda img, label: _one_hot_encode(img, label, num_classes))
    
    # Optional: Augment the data
    if augmentations is not None:
        assert isinstance(augmentations, int), 'The number of augmentations must be an integer'
        
        augmented_datasets = train_ds.flat_map(lambda img, label: _augment_multiple(img, label, augmentations, OneChannelOnly))
        train_ds = train_ds.concatenate(augmented_datasets)
    
    return test_ds, train_ds
