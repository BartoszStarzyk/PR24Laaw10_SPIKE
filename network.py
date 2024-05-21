import nengo
import nengo_dl
import tensorflow as tf
import numpy as np


def net():
    with nengo.Network(seed=0) as model:
        model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
        model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        model.config[nengo.Connection].synapse = None
        neuron_type = nengo.LIF(amplitude=0.01)

        nengo_dl.configure_settings(stateful=False)

        inp = nengo.Node(np.zeros(28 * 28))

        x = nengo_dl.Layer(
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")
        )(inp, shape_in=(28, 28, 1))
        x = nengo_dl.Layer(neuron_type)(x)

        x = nengo_dl.Layer(
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")
        )(x, shape_in=(26, 26, 32))
        x = nengo_dl.Layer(neuron_type)(x)

        x = nengo_dl.Layer(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(
            x, shape_in=(24, 24, 64)
        )
        x = nengo_dl.Layer(neuron_type)(x)

        x = nengo_dl.Layer(
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")
        )(x, shape_in=(12, 12, 64))
        x = nengo_dl.Layer(neuron_type)(x)

        # x = nengo_dl.Layer(
        #     tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")
        # )(x, shape_in=(10, 10, 128))
        # x = nengo_dl.Layer(neuron_type)(x)

        x = nengo_dl.Layer(tf.keras.layers.Flatten())(x, shape_in=(10, 10, 128))

        x = nengo_dl.Layer(tf.keras.layers.Dense(units=256, activation="relu"))(x)
        out = nengo_dl.Layer(tf.keras.layers.Dense(units=36))(x)

        # Probes to record the output
        out_p = nengo.Probe(out, label="out_p")
        out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")
        return model, out_p, out_p_filt


if __name__ == "__main__":
    a, b, c = net()
