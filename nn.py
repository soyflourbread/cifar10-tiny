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


class PreStem(tf.keras.Model):
    def __init__(self, prefix="prestem"):
        super(PreStem, self).__init__(name="{}".format(prefix))
        self.norm = tf.keras.layers.Normalization(
            mean=[
                0.4913997551666284 * 255,
                0.48215855929893703 * 255,
                0.4465309133731618 * 255
            ],
            variance=[
                (0.24703225141799082 * 255) ** 2,
                (0.24348516474564 * 255) ** 2,
                (0.26158783926049628 * 255) ** 2
            ],
            name="{}_norm".format(prefix),
        )

    def call(self, input_tensor, training=False):
        return self.norm(input_tensor)


class Stem(tf.keras.Model):
    def __init__(self, filter_count, prefix="stem"):
        super(Stem, self).__init__(name="{}".format(prefix))
        self.conv = tf.keras.layers.Conv2D(
            filter_count, 4, padding="same",
            name="{}-conv".format(prefix)
        )
        self.lnorm = tf.keras.layers.LayerNormalization(
            name="{}-layernorm".format(prefix)
        )

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        return self.lnorm(x)


class DownSample(tf.keras.Model):
    def __init__(self, filter_count, prefix="downsample"):
        super(DownSample, self).__init__(name="{}".format(prefix))
        self.lnorm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="{}-lnorm".format(prefix)
        )
        self.conv = tf.keras.layers.Conv2D(
            filter_count,
            kernel_size=2,
            strides=2,
            name="{}-conv".format(prefix)
        )

    def call(self, input_tensor, training=False):
        x = self.lnorm(input_tensor)
        return self.conv(x)


class DownSampleLight(tf.keras.Model):
    def __init__(self, filter_count, prefix="downsamplelight"):
        super(DownSampleLight, self).__init__(name="{}".format(prefix))
        self.dense = tf.keras.layers.Dense(
            filter_count,
            name="{}-dense".format(prefix)
        )
        self.pool = tf.keras.layers.MaxPool2D()
        self.lnorm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name="{}-lnorm".format(prefix)
        )

    def call(self, input_tensor, training=False):
        x = self.dense(input_tensor)
        x = self.pool(x)
        return self.lnorm(x)


class Bottleneck(tf.keras.Model):
    def __init__(self, filter_count, factor=4, switch_init_a=1e-6, prefix="bottleneck"):
        super(Bottleneck, self).__init__(name="{}".format(prefix))
        self.dconv = tf.keras.layers.DepthwiseConv2D(
            3, padding="same",
            depth_multiplier=factor,
            name="{}-dconv".format(prefix)
        )
        self.lnorm = tf.keras.layers.LayerNormalization(
            name="{}-layernorm".format(prefix)
        )
        self.dropout = tf.keras.layers.Dropout(
            0.5,
            name="{}-dropout".format(prefix)
        )
        self.dense = tf.keras.layers.Dense(
            filter_count,
            name="{}-dense-post".format(prefix)
        )
        self.scale = LayerScale(
            switch_init_a, filter_count,
            name="{}-switch".format(prefix)
        )

    def call(self, input_tensor, training=False):
        x = self.dconv(input_tensor)

        x = self.lnorm(x)
        x = tf.keras.activations.gelu(x)
        x = self.dropout(x)
        x = self.dense(x)

        return self.scale(x)


class DognetBlock(tf.keras.Model):
    def __init__(self, filter_count, layer_count, factor=4, prefix="dblock"):
        super(DognetBlock, self).__init__(name="{}".format(prefix))

        self.bottleneck_vec = [
            Bottleneck(
                filter_count, factor=factor,
                prefix="{}-bottleneck-{}".format(prefix, i)
            ) for i in range(layer_count)
        ]

        self.merge_vec = [
            tf.keras.layers.Add(
                name="{}-bottleneck-{}-merge".format(prefix, i)
            ) for i in range(layer_count)
        ]

    def call(self, input_tensor, training=False):
        x = input_tensor

        for bottleneck, merge in zip(self.bottleneck_vec, self.merge_vec):
            x_sub = x

            x = bottleneck(x)
            x = merge([x_sub, x])

        return x


class Head(tf.keras.Model):
    def __init__(self, num_classes, prefix="head"):
        super(Head, self).__init__(name="{}".format(prefix))
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.lnorm = tf.keras.layers.LayerNormalization(
            name="{}-layernorm".format(prefix)
        )
        self.relu = tf.keras.layers.ReLU(
            name="{}-activation".format(prefix)
        )
        self.dense = tf.keras.layers.Dense(
            num_classes,
            name="{}-dense".format(prefix)
        )

    def call(self, input_tensor, training=False):
        x = self.pool(input_tensor)

        x = self.lnorm(x)
        x = self.relu(x)

        return self.dense(x)


def create_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))

    x = PreStem(prefix="prestem")(inputs)

    x = Stem(32, prefix="stem")(x)

    x = DognetBlock(32, 2, factor=2, prefix="dog-1")(x)

    x = DownSampleLight(64, prefix="downsample-1")(x)
    x = DognetBlock(64, 4, factor=4, prefix="dog-2")(x)

    x = DownSampleLight(128, prefix="downsample-2")(x)
    x = DognetBlock(128, 2, factor=2, prefix="dog-3")(x)

    outputs = Head(10, prefix="head")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)
