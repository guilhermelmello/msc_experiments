import tensorflow as tf


def get_tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

    # config tpu client
    # from cloud_tpu_client import Client
    # c = Client(tpu='')
    # c.configure_tpu_version(tf.__version__, restart_type='ifNeeded')

    return strategy


def get_gpu_strategy():
    return tf.distribute.MirroredStrategy()


def get_strategy(device_type=None):
    if device_type is None:
        print("Detecting distributed strategy...")
        try:
            print("Trying TPU strategy...")
            strategy = get_tpu_strategy()
        except Exception:
            try:
                print("Trying GPU strategy...")
                strategy = get_gpu_strategy()
            except Exception:
                print("Trying default strategy...")
                strategy = tf.distribute.get_strategy()
    else:
        if device_type == "TPU":
            print("Selecting TPU strategy...")
            strategy = get_tpu_strategy()
        elif device_type == "GPU":
            print("Selecting GPU strategy...")
            strategy = get_gpu_strategy()
        else:
            print("Selecting default strategy...")
            strategy = tf.distribute.get_strategy()

    print("Using distributed strategy:", strategy)
    return strategy
