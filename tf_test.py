import tensorflow as tf

# Überprüfe die GPU-Verfügbarkeit
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))