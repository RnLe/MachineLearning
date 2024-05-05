# Facial image emotion recognition using CNN
# Target classes: Angry, Disgusted, Fearful, Happy, Sad, Surprised, Neutral

import tensorflow as tf

NUM_THREADS = 10

tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)

# DATASET
# files in dataset/test and dataset/train
# subfolders: angry, disgusted, fearful, happy, sad, surprised, neutral

# load dataset
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Test data
# Check for the number of images in each class and the size of the images
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 0 = alle Meldungen werden ausgegeben

# List of classes
classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# Number of images in each class
print('Number of training images in each class:')
for c in classes:
    path = os.path.join(train_dir, c)
    print(f'{c}: {len(os.listdir(path))} images')
    
print('\nNumber of test images in each class:')
for c in classes:
    path = os.path.join(test_dir, c)
    print(f'{c}: {len(os.listdir(path))} images')
    
# Image size
img = cv2.imread('dataset/train/angry/im0.png')
print(f'\nImage size: {img.shape}')

# CNN MODEL
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(48, 48, 3)),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.layers import Rescaling
import pathlib

# Anzahl der Klassen
num_classes = 7

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [48, 48])
    return img

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def encode_label(label, label_lookup):
    label_id = label_lookup(label)
    # Konvertiere die Vokabulargröße in int32
    depth = tf.cast(label_lookup.vocabulary_size(), tf.int32)
    return tf.one_hot(label_id, depth=depth)

# Erstelle den StringLookup Layer und passe ihn an
def prepare_label_lookup(train_dir):
    # Erstelle ein Dataset von Dateipfaden
    file_paths = tf.data.Dataset.list_files(str(pathlib.Path(train_dir) / '*/*'), shuffle=False)
    
    # Extrahiere Labels aus den Pfaden
    labels = file_paths.map(lambda x: tf.strings.split(x, os.path.sep)[-2])
    
    # Nutze den StringLookup Layer, um die Labels zu indizieren
    label_lookup = tf.keras.layers.StringLookup(num_oov_indices=0)
    label_lookup.adapt(labels)
    
    return label_lookup

label_lookup = prepare_label_lookup(train_dir)

def process_path(file_path, label_lookup):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    label = encode_label(label, label_lookup)
    return img, label

# Dataset-Erstellung und Mapping
train_ds = tf.data.Dataset.list_files(str(pathlib.Path(train_dir) / '*/*.png'), shuffle=False)
train_ds = train_ds.map(lambda x: process_path(x, label_lookup), num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_ds = tf.data.Dataset.list_files(str(pathlib.Path(test_dir) / '*/*.png'), shuffle=False)
test_ds = test_ds.map(lambda x: process_path(x, label_lookup), num_parallel_calls=tf.data.experimental.AUTOTUNE)

cache_dir = 'cache'

train_ds = train_ds.shuffle(640).batch(64).prefetch(tf.data.experimental.AUTOTUNE).cache()
test_ds = test_ds.shuffle(640).batch(64).prefetch(tf.data.experimental.AUTOTUNE).cache()

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='models/model.h5',
    save_best_only=True,
    monitor='val_loss',  # Überwachte Metrik
    mode='min',  # 'min' für Metriken, die minimiert werden sollen, 'max' für maximierende Metriken
    save_weights_only=False,  # True, wenn nur Gewichte gespeichert werden sollen, False für das gesamte Modell
    verbose=1
)

history = model.fit(
    train_ds,
    # Show all logs
    verbose=1,
    epochs=100,
    validation_data=test_ds,
    callbacks=[checkpoint_callback]
)

# PLOT TRAINING AND VALIDATION ACCURACY
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('accuracy.png')
plt.show()

# PLOT TRAINING AND VALIDATION LOSS
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 2])
plt.legend(loc='upper right')
plt.savefig('loss.png')
plt.show()

# SAVE MODEL
model.save('emotion_recognition_model.h5')
print('Model saved as emotion_recognition_model.h5')

# TEST MODEL
# Load model
model = tf.keras.models.load_model('emotion_recognition_model.h5')

# Load test image
img = cv2.imread('dataset/test/angry/im0.png')
img = cv2.resize(img, (48, 48))
img = np.reshape(img, [1, 48, 48, 3])

# Predict emotion
prediction = model.predict(img)
emotion = classes[np.argmax(prediction)]
print(f'Predicted emotion: {emotion}')

# Display image
plt.imshow(cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f'Predicted emotion: {emotion}')
plt.savefig('test_image.png')
plt.show()

# Display prediction
plt.bar(classes, prediction[0])
plt.ylabel('Probability')
plt.title('Emotion prediction')
plt.savefig('prediction.png')
plt.show()