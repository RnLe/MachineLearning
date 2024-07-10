import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class DenseBlock(layers.Layer):
    def __init__(
        self,
        num_layers,
        growth_rate,
        dropout_rate=0.0,
        l2_regularization=1e-4,
        name=None,
    ):
        super(DenseBlock, self).__init__(name=name)
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization
        self.layers_list = []

        for i in range(num_layers):
            self.layers_list.append(self._make_layer(growth_rate, name=f"layer_{i+1}"))

    def _make_layer(self, growth_rate, name=None):
        layers_seq = [
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(
                4 * growth_rate,
                kernel_size=1,
                use_bias=False,
                kernel_regularizer=regularizers.l2(self.l2_regularization),
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(
                growth_rate,
                kernel_size=3,
                padding="same",
                use_bias=False,
                kernel_regularizer=regularizers.l2(self.l2_regularization),
            ),
        ]
        if self.dropout_rate > 0:
            layers_seq.append(layers.Dropout(self.dropout_rate))
        return tf.keras.Sequential(layers_seq, name=name)

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            output = layer(x)
            x = tf.concat([x, output], axis=-1)
        return x


class TransitionLayer(layers.Layer):
    def __init__(self, reduction, l2_regularization=1e-4, name=None):
        super(TransitionLayer, self).__init__(name=name)
        self.reduction = reduction
        self.l2_regularization = l2_regularization

    def build(self, input_shape):
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv = layers.Conv2D(
            int(input_shape[-1] * self.reduction),
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=regularizers.l2(self.l2_regularization),
        )
        if input_shape[1] > 2 and input_shape[2] > 2:
            self.avg_pool = layers.AveragePooling2D(pool_size=2, strides=2)
        else:
            self.avg_pool = None

    def call(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        x = self.conv(x)
        #if self.avg_pool is not None:
        #    x = self.avg_pool(x)
        if self.avg_pool and x.shape[1] > 1 and x.shape[2] > 1:
            x = self.avg_pool(x)
        return x


class DenseNet(Model):
    def __init__(
        self,
        num_blocks=4,
        num_layers_per_block=[6, 12, 24, 16],
        growth_rate=32,
        reduction=0.5,
        num_classes=7,
        input_shape=(224, 224, 3),
        dropout_rate=0.5,
        l2_regularization=1e-4,
    ):
        """
        Initializes a DenseNet model.

        Args:
        num_blocks (int): The number of dense blocks in the network.
        num_layers_per_block (list of int): List specifying the number of layers in each dense block.
        growth_rate (int): The number of filters to add per dense block layer.
        reduction (float): The reduction factor applied through the transition layers.
        num_classes (int): The number of classes for the final classification layer (i.e. the model output).
        input_shape (tuple): The shape of the input images (height, width, channels).
        dropout_rate (float): The dropout rate used in each dense block.
        l2_regularization (float): Coefficient for L2 regularization applied to all convolutional layers.
        """
        super(DenseNet, self).__init__()
        self.num_blocks = num_blocks
        self.growth_rate = growth_rate
        self.reduction = reduction
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.l2_regularization = l2_regularization

        self.initial_conv = layers.Conv2D(
            2 * growth_rate,
            kernel_size=7,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(l2_regularization),
        )
        self.initial_bn = layers.BatchNormalization()
        self.initial_relu = layers.ReLU()
        self.initial_pool = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")

        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(
                DenseBlock(
                    num_layers_per_block[i],
                    growth_rate,
                    dropout_rate,
                    l2_regularization,
                    name=f"dense_block_{i+1}",
                )
            )
            if i != num_blocks - 1:
                self.blocks.append(
                    TransitionLayer(
                        reduction, l2_regularization, name=f"transition_layer_{i+1}"
                    )
                )

        self.final_bn = layers.BatchNormalization()
        self.final_relu = layers.ReLU()
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=regularizers.l2(l2_regularization),
        )

    def call(self, inputs):
        x = self.initial_conv(inputs)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        x = self.initial_pool(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        return x
