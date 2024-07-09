import densenet


def test_DenseNet():
    # Example usage
    input_shape = (224, 224, 3)
    num_blocks = 4
    num_layers_per_block = [6, 12, 24, 16]  # DenseNet-121 configuration
    growth_rate = 32
    reduction = 0.5
    num_classes = 7

    model = densenet.DenseNet(
        num_blocks,
        num_layers_per_block,
        growth_rate,
        reduction,
        num_classes,
        input_shape,
    )
    model.build((None, *input_shape))
    model.compile(
        optimizer=optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model(tf.keras.Input(input_shape))
    model.summary()
