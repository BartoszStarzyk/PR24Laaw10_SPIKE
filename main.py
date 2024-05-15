import nengo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# from urllib.request import urlretrieve
from nengo_dl import configure_settings, Layer, Simulator
from segmention import segmentoutletters

# (train_images, train_labels), (test_images, test_labels) = (
#     tf.keras.datasets.mnist.load_data()
# )

# train_images = train_images.reshape((train_images.shape[0], -1))
# test_images = test_images.reshape((test_images.shape[0], -1))


with nengo.Network(seed=0) as model:
    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
    model.config[nengo.Connection].synapse = None
    neuron_type = nengo.LIF(amplitude=0.01)

    configure_settings(stateful=False)

    inp = nengo.Node(np.zeros(28 * 28))

    x = Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(
        inp, shape_in=(28, 28, 1)
    )
    x = Layer(neuron_type)(x)

    x = Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))(
        x, shape_in=(26, 26, 32)
    )
    x = Layer(neuron_type)(x)

    x = Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))(
        x, shape_in=(12, 12, 64)
    )
    x = Layer(neuron_type)(x)

    out = Layer(tf.keras.layers.Dense(units=10))(x)

    out_p = nengo.Probe(out, label="out_p")
    out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")

# minibatch_size = 200
minibatch_size = 10
sim = Simulator(model, minibatch_size=minibatch_size)

# train_images = train_images[:, None, :]
# train_labels = train_labels[:, None, None]

# n_steps = 120
# test_images = np.tile(test_images[:, None, :], (1, n_steps, 1))
# test_labels = np.tile(test_labels[:, None, None], (1, n_steps, 1))


def classification_accuracy(y_true, y_pred):
    return tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])


sim.compile(loss={out_p_filt: classification_accuracy})

# load parameters
sim.load_params("./mnist_params")
# data = sim.predict(test_images[:minibatch_size])
# our_images = load_images()

plot = True
for name in os.listdir("../Dane"):
    our_images, im = segmentoutletters(name)
    data = sim.predict(our_images)

    label_text = ""
    for i in range(10):
        if plot:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(our_images[i, 0].reshape((28, 28)), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.plot(tf.nn.softmax(data[out_p_filt][i]))
            plt.legend([str(i) for i in range(10)], loc="upper left")
            plt.xlabel("timesteps")
            plt.ylabel("probability")
            plt.tight_layout()
        label_text += str(np.argmax(tf.nn.softmax(data[out_p_filt][i])[-1, :]))

    plt.figure()
    plt.imshow(im)
    plt.title(label_text, fontsize=40)
    plt.axis("off")

plt.show()
sim.close()
