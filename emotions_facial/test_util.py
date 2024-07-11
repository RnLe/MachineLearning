from util import evaluate, load_data
import tensorflow as tf
from densenet import *


# XXX: re-save the model with the added @saving hooks
#def test_evaluate():
#    classes = sorted(["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"])
#    model = tf.keras.models.load_model("test_model.h5")
#    x_test, y_test, y_test_encoded = load_data(classes)
#    evaluate(model, x_test, y_test, y_test_encoded, classes)
