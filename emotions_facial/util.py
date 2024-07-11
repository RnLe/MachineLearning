import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def plot_history(history, modelName=""):
    highest_val_acc = np.argmax(history.history["val_accuracy"])
    highest_val_acc_item = (
        highest_val_acc,
        history.history["val_accuracy"][highest_val_acc],
    )

    lowest_val_loss = np.argmin(history.history["val_loss"])
    lowest_val_loss_item = (
        lowest_val_loss,
        history.history["val_loss"][lowest_val_loss],
    )

    size = 5
    plt.figure(figsize=(2 * size, size))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], c="C0", label="train")
    plt.plot(history.history["val_loss"], c="C1", label="test")
    plt.plot(
        lowest_val_loss_item[0],
        lowest_val_loss_item[1],
        ".",
        markersize=20,
        color="C2",
        label=f"Lowest val loss = {lowest_val_loss_item[1]:.3f} ",
    )
    plt.title(modelName + " Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], c="C0", label="train")
    plt.plot(history.history["val_accuracy"], c="C1", label="test")
    plt.plot(
        highest_val_acc_item[0],
        highest_val_acc_item[1],
        ".",
        markersize=20,
        color="C2",
        label=f"Highest val acc = {highest_val_acc_item[1]:.3f}",
    )
    plt.title(modelName + " Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.grid()
    plt.legend()
    plt.show()


def load_data(directory, classes, greyscale=True):
    """load_data loads data as np.array, not as tensorflow abstract tensor
    """
    # Load test data
    x = []
    y = []
    for i, c in enumerate(classes):
        path = os.path.join(directory, c)
        for img_name in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_name))
            if img is not None:  # Ensure the image was read correctly
                if greyscale:
                    img = cv2.resize(img, (48, 48))
                    # Image to greyscale and shape (48, 48, 1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(
                        img, axis=-1
                    )  # Add channel dimension for greyscale images
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Normalize the image
                img = img / 255.0
                x.append(img)
                y.append(i)
    x = np.array(x)
    y = np.array(y)
    y_encoded = to_categorical(y, num_classes=len(classes))
    return x, y, y_encoded


def create_class_weights(y_train_encoded):
    y_train_labels = np.argmax(y_train_encoded, axis=1)
    y_train_labels = y_train_labels.astype(np.int32)
    # Berechnung der Klassengewichte
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train_labels), y=y_train_labels
    )
    # Convert the class weights to a dictionary. Use np.int32 as the key type
    class_weights_dict = {
        int(k): float(v) for k, v in zip(np.unique(y_train_labels), class_weights)
    }
    return class_weights_dict


def evaluate(model, x_test, y_test, y_test_encoded, classes):
    # Evaluate model
    evaluation = model.evaluate(x_test, y_test_encoded)
    print(evaluation)

    # Predict test data
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Normalize the confusion matrix
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    print("Confusion Matrix:\n", cm)
    print("Normalized Confusion Matrix:\n", cm_normalized)
    # Display normalized confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.show()

    # 5x5 matrix with random pictures from the test set, labeled with the predicted emotion and the true emotion; text color is green if the prediction is correct, red otherwise
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        index = random.randint(0, len(x_test) - 1)
        img = x_test[index]
        img = np.reshape(img, [1, 48, 48, 1])
        prediction = model.predict(img)
        emotion = classes[np.argmax(prediction)]
        true_emotion = classes[y_test[index]]
        color = "g" if emotion == true_emotion else "r"
        # Image back to original size
        img = x_test[index]
        img = np.reshape(img, [48, 48])
        plt.imshow(img, cmap="gray")
        plt.title(f"Pred: {emotion}\nTrue: {true_emotion}", color=color)
        plt.axis("off")
        plt.tight_layout()
    plt.show()
