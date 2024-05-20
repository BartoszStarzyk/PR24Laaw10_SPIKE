import nengo
import nengo_dl
import tensorflow as tf
import numpy as np

# with nengo.Network(seed=0) as model:
#     model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
#     model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
#     model.config[nengo.Connection].synapse = None
#     neuron_type = nengo.LIF(amplitude=0.01)

#     configure_settings(stateful=False)

#     inp = nengo.Node(np.zeros(28 * 28))

#     x = Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(
#         inp, shape_in=(28, 28, 1)
#     )
#     x = Layer(neuron_type)(x)

#     x = Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))(
#         x, shape_in=(26, 26, 32)
#     )
#     x = Layer(neuron_type)(x)

#     x = Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))(
#         x, shape_in=(12, 12, 64)
#     )
#     x = Layer(neuron_type)(x)

#     out = Layer(tf.keras.layers.Dense(units=36))(x)

#     out_p = nengo.Probe(out, label="out_p")
#     out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")


#     with nengo.Network(seed=0) as model:
#         model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
#         model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
#         model.config[nengo.Connection].synapse = None
#         neuron_type = nengo.LIF(amplitude=0.01)

#         configure_settings(stateful=False)

#         inp = nengo.Node(np.zeros(28 * 28))

#         # First convolutional layer
#         x = Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))(
#             inp, shape_in=(28, 28, 1)
#         )
#         x = Layer(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(
#             x, shape_in=(26, 26, 32)
#         )
#         x = Layer(neuron_type)(x)

#         # Second convolutional layer
#         x = Layer(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))(
#             x, shape_in=(13, 13, 32)
#         )
#         x = Layer(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(
#             x, shape_in=(11, 11, 64)
#         )
#         x = Layer(neuron_type)(x)

#         # Third convolutional layer
#         x = Layer(
#             tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")
#         )(x, shape_in=(5, 5, 64))
#         x = Layer(neuron_type)(x)

#         # Flatten layer
#         x = Layer(tf.keras.layers.Flatten())(x, shape_in=(3, 3, 128))

#         # Dense layer before output
#         x = Layer(tf.keras.layers.Dense(units=256, activation="relu"))(x)
#         x = Layer(neuron_type)(x)

#         # Dense output layer with units matching EMNIST classes (e.g., 47 for EMNIST Letters)
#         out = Layer(tf.keras.layers.Dense(units=36))(x)

#         # Probes to record the output
#         out_p = nengo.Probe(out, label="out_p")
#         out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")

# # chyba drukowane
#     with nengo.Network(seed=0) as model:
#         model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
#         model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
#         model.config[nengo.Connection].synapse = None
#         neuron_type = nengo.LIF(amplitude=0.01)

#         configure_settings(stateful=False)

#         # Input node for EMNIST images (28x28 pixels)
#         inp = nengo.Node(np.zeros(28 * 28))

#         # First convolutional layer
#         x = Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))(
#             inp, shape_in=(28, 28, 1)
#         )
#         x = Layer(neuron_type)(x)

#         # Second convolutional layer
#         x = Layer(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"))(
#             x, shape_in=(26, 26, 32)
#         )
#         x = Layer(neuron_type)(x)

#         # Third convolutional layer with stride
#         x = Layer(
#             tf.keras.layers.Conv2D(
#                 filters=64, strides=2, kernel_size=3, activation="relu"
#             )
#         )(x, shape_in=(24, 24, 64))
#         x = Layer(neuron_type)(x)

#         # Fourth convolutional layer
#         x = Layer(
#             tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")
#         )(x, shape_in=(11, 11, 64))
#         x = Layer(neuron_type)(x)

#         # Fifth convolutional layer with stride
#         x = Layer(
#             tf.keras.layers.Conv2D(
#                 filters=128, strides=2, kernel_size=3, activation="relu"
#             )
#         )(x, shape_in=(9, 9, 128))
#         x = Layer(neuron_type)(x)

#         # Flatten layer
#         x = Layer(tf.keras.layers.Flatten())(x, shape_in=(4, 4, 128))

#         # Dense layer before output
#         x = Layer(tf.keras.layers.Dense(units=256, activation="relu"))(x)
#         x = Layer(neuron_type)(x)

#         # Dense output layer with units matching EMNIST classes (e.g., 47 for EMNIST Letters)
#         out = Layer(tf.keras.layers.Dense(units=36))(x)

#         out_p = nengo.Probe(out, label="out_p")
#         out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")
#     return model, out_p_filt


####
####
####
####
# PARAMETERS.npz, ostatnia działająza mnistowa architektura
def net():
    with nengo.Network(seed=0) as model:
        model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
        model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
        model.config[nengo.Connection].synapse = None
        neuron_type = nengo.LIF(amplitude=0.01)

        nengo_dl.configure_settings(stateful=False)

        inp = nengo.Node(np.zeros(28 * 28))

        # First convolutional layer
        x = nengo_dl.Layer(
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")
        )(inp, shape_in=(28, 28, 1))
        x = nengo_dl.Layer(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(
            x, shape_in=(26, 26, 32)
        )
        x = nengo_dl.Layer(neuron_type)(x)

        # Second convolutional layer
        x = nengo_dl.Layer(
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")
        )(x, shape_in=(13, 13, 32))
        x = nengo_dl.Layer(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(
            x, shape_in=(11, 11, 64)
        )
        x = nengo_dl.Layer(neuron_type)(x)

        # Third convolutional layer
        x = nengo_dl.Layer(
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")
        )(x, shape_in=(5, 5, 64))
        x = nengo_dl.Layer(neuron_type)(x)

        # Flatten layer
        x = nengo_dl.Layer(tf.keras.layers.Flatten())(x, shape_in=(3, 3, 128))

        # Dense layer before output
        x = nengo_dl.Layer(tf.keras.layers.Dense(units=256, activation="relu"))(x)
        x = nengo_dl.Layer(neuron_type)(x)

        # Dense output layer with units matching EMNIST classes (e.g., 47 for EMNIST Letters)
        out = nengo_dl.Layer(tf.keras.layers.Dense(units=36))(x)

        # Probes to record the output
        out_p = nengo.Probe(out, label="out_p")
        out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")
        return model, out_p, out_p_filt


###########################################################
# def net():
#     with nengo.Network(seed=0) as model:
#         model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
#         model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
#         model.config[nengo.Connection].synapse = None
#         neuron_type = nengo.LIF(amplitude=0.01)

#         nengo_dl.configure_settings(stateful=False)

#         inp = nengo.Node(np.zeros(28 * 28))

#         x = nengo_dl.Layer(
#             tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")
#         )(inp, shape_in=(28, 28, 1))
#         x = nengo_dl.Layer(neuron_type)(x)

#         x = nengo_dl.Layer(
#             tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")
#         )(x, shape_in=(26, 26, 32))
#         x = nengo_dl.Layer(neuron_type)(x)

#         x = nengo_dl.Layer(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))(
#             x, shape_in=(24, 24, 64)
#         )
#         x = nengo_dl.Layer(neuron_type)(x)

#         x = nengo_dl.Layer(
#             tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")
#         )(x, shape_in=(12, 12, 64))
#         x = nengo_dl.Layer(neuron_type)(x)

#         # x = nengo_dl.Layer(
#         #     tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")
#         # )(x, shape_in=(10, 10, 128))
#         # x = nengo_dl.Layer(neuron_type)(x)

#         x = nengo_dl.Layer(tf.keras.layers.Flatten())(x, shape_in=(10, 10, 128))

#         x = nengo_dl.Layer(tf.keras.layers.Dense(units=256, activation="relu"))(x)
#         out = nengo_dl.Layer(tf.keras.layers.Dense(units=36))(x)

#         # Probes to record the output
#         out_p = nengo.Probe(out, label="out_p")
#         out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")
#         return model, out_p, out_p_filt


if __name__ == "__main__":
    a, b = net()
