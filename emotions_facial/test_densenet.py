import densenet
import tensorflow as tf
from tensorflow.keras import optimizers


def test_DenseNet():
    """
    Test the DenseNet model by building, compiling, and summarizing it.
    This function uses predefined settings for a DenseNet configuration.
    """
    # Set up model parameters with default arguments
    input_shape = (224, 224, 3)
    num_blocks = 4
    num_layers_per_block = [6, 12, 24, 16]  # DenseNet-121 configuration
    growth_rate = 32
    reduction = 0.5
    num_classes = 7
    dropout_rate = 0.5  # Adjust dropout rate as needed
    l2_regularization = 1e-4  # Adjust L2 weight decay as needed

    # Initialize and build the DenseNet model
    model = densenet.DenseNet(
        num_blocks=num_blocks,
        num_layers_per_block=num_layers_per_block,
        growth_rate=growth_rate,
        reduction=reduction,
        num_classes=num_classes,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        l2_regularization=l2_regularization,
    )

    model.build((None, *input_shape))  # Build the model to initialize weights

    # Compile the model with optimization and loss settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Create a dummy input and pass it through the model to check summary
    dummy_input = tf.keras.Input(shape=input_shape)
    model(dummy_input)
    model.summary()  # Display the model architecture
