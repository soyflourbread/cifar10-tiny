import tensorflow as tf


class LayerScale(tf.keras.layers.Layer):
    def __init__(self, init_values, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim

    def build(self, input_shape):
        self.gamma = tf.Variable(
            self.init_values * tf.ones((self.projection_dim,), dtype=self.compute_dtype),
            dtype=self.compute_dtype
        )

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "init_values": self.init_values,
                "projection_dim": self.projection_dim,
            }
        )
        return config


def bottleneck(
        filter_count, factor,
        kernel_size,
        prefix="bottleneck",
        switch_init_a=1e-6
):
    def _bottleneck_impl(x):
        # x = tf.keras.layers.DepthwiseConv2D(
        #     3, padding="same",
        #     name="{}-dconv".format(prefix)
        # )(x)
        x = tf.keras.layers.DepthwiseConv2D(
            3, padding="same",
            depth_multiplier=factor,
            name="{}-dconv".format(prefix)
        )(x)
        x = tf.keras.layers.LayerNormalization(
            name="{}-layernorm".format(prefix)
        )(x)
        x = tf.keras.activations.gelu(x)
        # x = tf.keras.layers.Dropout(
        #     0.2,
        #     name="{}-dropout".format(prefix)
        # )(x)
        x = tf.keras.layers.Dense(
            filter_count,
            name="{}-dense".format(prefix)
        )(x)
        x = LayerScale(
            switch_init_a, filter_count,
            name="{}-switch".format(prefix)
        )(x)

        return x

    return _bottleneck_impl


def dognet_block(
        filter_count,
        layer_count,
        kernel_size,
        prefix="dblock",
        factor=4
):
    def _dognet_block_impl(x):
        x_main = x
        for i in range(layer_count):
            x_sub = x_main

            x_main = bottleneck(
                filter_count, factor, kernel_size,
                prefix="{}-bottleneck-{}".format(prefix, i)
            )(x_main)
            x_main = tf.keras.layers.Add(
                name="{}-bottleneck-{}-merge".format(prefix, i)
            )([x_sub, x_main])

        return x_main

    return _dognet_block_impl


def create_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))

    x = tf.keras.layers.Conv2D(32, 4, padding="same")(inputs)
    x = dognet_block(32, 2, 3, factor=4, prefix="dog-1")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Dense(64)(x)
    x = dognet_block(64, 4, 3, factor=4, prefix="dog-2")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Dense(128)(x)
    x = dognet_block(128, 2, 3, factor=2, prefix="dog-3")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    outputs = tf.keras.layers.Dense(10)(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
