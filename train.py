import tensorflow as tf

from nn import create_model
from dataset import fetch_dataset

import datetime

import argparse


def enable_mixed():
    print("Warn: enabling mixed precision")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')


def configure_tf():
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices) > 0:
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=10000)])


def configure_tf_tpu():
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)

    return strategy


def create_and_backpropagate(ds_train, ds_test, epoch):
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
                  )
    model.summary()

    model_dir = "model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    try:
        model.fit(
            ds_train,
            epochs=epoch,
            validation_data=ds_test,
            callbacks=[tensorboard_callback]
        )
    except KeyboardInterrupt:
        print("Interrupt received. Saving model...")

    model.save(model_dir)


def run(batch_size, augment):
    configure_tf()

    ds_train, ds_test = fetch_dataset(batch_size, augment)

    create_and_backpropagate(ds_train, ds_test, 200)


def run_tpu(batch_size, augment):
    print("Running from TPU...")

    strategy = configure_tf_tpu()

    ds_train, ds_test = fetch_dataset(batch_size, augment)

    with strategy.scope():
        create_and_backpropagate(ds_train, ds_test, 200)


def main():
    parser = argparse.ArgumentParser(
        prog="dognet-train",
        description='it does',
        epilog='something')
    parser.add_argument('-b', '--batch', type=int)
    parser.add_argument('-a', '--augment', action='store_true')
    parser.add_argument('-t', '--tpu', action='store_true')
    parser.add_argument('-m', '--mixed', action='store_true')
    args = parser.parse_args()

    print("Running with config: [batch={}, augment={}]".format(args.batch, args.augment))

    if args.mixed:
        enable_mixed()

    if args.tpu:
        run_tpu(args.batch, args.augment)
    else:
        run(args.batch, args.augment)


if __name__ == "__main__":
    main()
