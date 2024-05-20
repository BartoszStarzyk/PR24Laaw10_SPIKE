import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from string import ascii_uppercase
import network
from nengo_dl import Simulator
from segmention import segmentoutletters

model, out_p, out_p_filt = network.net()
minibatch_size = 10
sim = Simulator(model, minibatch_size=minibatch_size)


def classification_accuracy(y_true, y_pred):
    return tf.metrics.sparse_categorical_accuracy(y_true[:, -1], y_pred[:, -1])


def test_accuracy(label, prediction):
    labels = {
        "(BARCODE)0003": "AT_02001/2",
        "(BARCODE)0007": "AT_02001/6",
        "(BARCODE)0015": "AT_02002/4",
        "BG_0005": "BG_01001/5",
        "BG_0012": "BG_01002/2",
        "BG_0014": "BG_01002/4",
    }
    return sum([pred == lab for pred, lab in zip(prediction, labels[label])]) / 10


sim.load_params("./previous_weights/mnist_params")
sim.compile(loss={out_p_filt: classification_accuracy})


accuracies = 0
a = list(map(str, range(36)))
b = list(map(str, range(10))) + list(ascii_uppercase)
convert_classes = dict(zip(a, b))

plot = True
plot_indices = [0]
plot_index = 0

for name in os.listdir("../Dane"):
    our_images, im = segmentoutletters(name)
    data = sim.predict(our_images)
    # data = sim.predict(test_images[:minibatch_size])
    labal_label = name[:-4]
    label_text = ""
    for i in range(10):
        if plot and plot_index in plot_indices:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(our_images[i, 0].reshape((28, 28)), cmap="gray")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.plot(tf.nn.softmax(data[out_p_filt][i]))
            plt.legend(
                [str(i) for i in range(10)] + list(ascii_uppercase), loc="upper left"
            )
            plt.xlabel("timesteps")
            plt.ylabel("probability")
        label_text += convert_classes[str(np.argmax(data[out_p_filt][i][-1, :]))]
    accuracy = test_accuracy(labal_label, label_text)
    accuracies += accuracy
    print(f"label {labal_label} accuracy: {accuracy * 100}%")
    plt.figure()
    plt.imshow(im)
    plt.title(label_text, fontsize=40)
    plt.axis("off")
    plot_index += 1
print(f"total accuracy: {accuracies / 6 * 100}%")
plt.show()
sim.close()

# pytjaniki jeżeli za mała pewność
# odległość najlepszego rozwiązania od średniej lub kolejnego
# zbadać wsparcie dla opencla przy nengo
