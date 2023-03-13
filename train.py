import tensorflow as tf

from nn import create_model
from dataset import fetch_dataset

import datetime


def configure_tf():
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices))

    if len(physical_devices) > 0:
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=10000)])


def main():
    configure_tf()

    ds_train, ds_test = fetch_dataset()

    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
                  )
    model.summary()

    model_dir = "model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model.fit(
        ds_train,
        epochs=100,
        validation_data=ds_test
    )

    model.save(model_dir)


if __name__ == "__main__":
    main()
