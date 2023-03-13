import tensorflow as tf
import tensorflow_datasets as tfds


def norm_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def augment_img(image, label):
    image = tf.keras.layers.RandomFlip("horizontal")(image)
    image = tf.keras.layers.RandomTranslation(
        0.1, 0.1,
        fill_mode="nearest",
        interpolation="nearest"
    )(image)
    return image, label


def fetch_dataset(batch_size, augment):
    (ds_train, ds_test) = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True
    )

    ds_train = ds_train.map(
        norm_img, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds_train = ds_train.map(
            augment_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.cache()
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        norm_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_test
