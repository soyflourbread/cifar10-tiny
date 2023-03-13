import tensorflow as tf

from nn import create_model, create_model_large
from dataset import fetch_dataset

import datetime

import argparse


def configure_tf():
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices) > 0:
        tf.config.set_logical_device_configuration(
            physical_devices[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=10000)])


def configure_tf_tpu():
    # tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)

    return strategy


def create_and_backpropagate(ds_train, ds_test, epoch, create_mode_func):
    model = create_mode_func()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
                  )
    model.summary()

    model_dir = "model/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    try:
        model.fit(
            ds_train,
            epochs=epoch,
            validation_data=ds_test
        )
    except KeyboardInterrupt:
        print("Interrupt received. Saving model...")

    model.save(model_dir)


def run(batch_size):
    configure_tf()

    ds_train, ds_test = fetch_dataset(batch_size)

    create_and_backpropagate(ds_train, ds_test, 100, create_model)


def run_tpu(batch_size):
    print("Running from TPU...")

    strategy = configure_tf_tpu()

    ds_train, ds_test = fetch_dataset(batch_size)

    with strategy.scope():
        create_and_backpropagate(ds_train, ds_test, 200, create_model_large)


def main():
    parser = argparse.ArgumentParser(
        prog="dognet-train",
        description='it does',
        epilog='something')
    parser.add_argument('-t', '--tpu', action='store_true')
    parser.add_argument('-b', '--batch', type=int)
    args = parser.parse_args()

    if args.tpu:
        run_tpu(args.batch)
    else:
        run(args.batch)


if __name__ == "__main__":
    main()
