import tensorflow as tf
from tensorflow.keras import layers, Model


class DenseBlock(layers.Layer):
    def __init__(self, num_layers, growth_rate, name=None):
        super(DenseBlock, self).__init__(name=name)
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.layers_list = []

        for i in range(num_layers):
            self.layers_list.append(self._make_layer(growth_rate, name=f"layer_{i+1}"))

    def _make_layer(self, growth_rate, name=None):
        return tf.keras.Sequential(
            [
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(4 * growth_rate, kernel_size=1, use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(
                    growth_rate, kernel_size=3, padding="same", use_bias=False
                ),
            ],
            name=name,
        )

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            output = layer(x)
            x = tf.concat([x, output], axis=-1)
        return x


class TransitionLayer(layers.Layer):
    def __init__(self, reduction, name=None):
        super(TransitionLayer, self).__init__(name=name)
        self.reduction = reduction

    def build(self, input_shape):
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv = layers.Conv2D(
            int(input_shape[-1] * self.reduction), kernel_size=1, use_bias=False
        )
        self.avg_pool = layers.AveragePooling2D(pool_size=2, strides=2)

    def call(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avg_pool(x)
        return x


class DenseNet(Model):
    def __init__(
        self,
        num_blocks,
        num_layers_per_block,
        growth_rate,
        reduction,
        num_classes,
        input_shape,
    ):
        super(DenseNet, self).__init__()
        self.num_blocks = num_blocks
        self.num_layers_per_block = num_layers_per_block
        self.growth_rate = growth_rate
        self.reduction = reduction
        self.num_classes = num_classes
        self.input_shape_ = input_shape

        self.initial_conv = layers.Conv2D(
            2 * growth_rate, kernel_size=7, strides=2, padding="same", use_bias=False
        )
        self.initial_bn = layers.BatchNormalization()
        self.initial_relu = layers.ReLU()
        self.initial_pool = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")

        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(
                DenseBlock(
                    num_layers_per_block[i], growth_rate, name=f"dense_block_{i+1}"
                )
            )
            if i != num_blocks - 1:
                self.blocks.append(
                    TransitionLayer(reduction, name=f"transition_layer_{i+1}")
                )

        self.final_bn = layers.BatchNormalization()
        self.final_relu = layers.ReLU()
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes, activation="softmax")

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
