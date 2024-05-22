# TODO: Podmienienie na OpenCLa, pomiary (też w labie)
# TODO: Wymyślenie co jest źle z ewaluacją (Pewnie coś z labelami ew.)
# TODO: Zapisywanie całej historii trenowania
# TODO: sprawdzenie wszystkich zapisanych wag
# TODO: Probe'y do wyresów?

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from string import ascii_uppercase
import network
from nengo_dl import Simulator
from segmention import segmentoutletters, SLASH_ID, UNDERSCORE_ID, QUESTIONMARK_ID


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


def plot_predictions(data, our_images, i, name):
    plt.title(name)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(our_images[i, 0].reshape((28, 28)), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.plot(data[i])
    plt.legend([str(i) for i in range(10)] + list(ascii_uppercase), loc="upper left")
    plt.xlabel("timesteps")
    plt.ylabel("probability")


def plot_results(image, labels, confidence):
    plt.figure(figsize=(6, 6))
    plt.title(labels, fontsize=40)
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.subplot(2, 1, 2)
    plt.bar(range(10), list(confidence))
    plt.axis("off")


k = list(map(str, range(36))) + [
    str(UNDERSCORE_ID),
    str(SLASH_ID),
    str(QUESTIONMARK_ID),
]
v = list(map(str, range(10))) + list(ascii_uppercase) + ["_", "/", "?"]
convert_classes = dict(zip(k, v))


model, out_p, out_p_filt = network.net()
sim = Simulator(model, minibatch_size=10)
sim.load_params("./wagi_epoki/params_epoch_11_dropout_123")
sim.compile(loss={out_p_filt: classification_accuracy})


accuracies = 0

plot_indices = [0]

for idx, name in enumerate(os.listdir("../Dane")):
    our_images, im, s_chars = segmentoutletters(
        name, save=False, detect_special_chars=True
    )
    data = tf.nn.softmax(sim.predict(our_images)[out_p_filt])
    predictions = np.argsort(data[:, -1, :], axis=1)
    prediction = predictions[:, -1]
    second_prediction = predictions[:, -2]
    # print(name, prediction, second_prediction)
    confidence = np.abs(
        data[:, -1, :].numpy()[np.arange(10), prediction]
        - data[:, -1, :].numpy()[np.arange(10), second_prediction]
    )
    prediction[s_chars != 0] = s_chars[s_chars != 0]
    label_label = name[:-4]
    label_text = ""
    for i in range(10):
        if idx in plot_indices:
            plot_predictions(data, our_images, i, name)
        label_text += convert_classes[str(prediction[i])]
    accuracy = test_accuracy(label_label, label_text)
    accuracies += accuracy
    print(f"label {label_label} accuracy: {accuracy * 100}%")
    plot_results(im, label_text, confidence)
print(f"total accuracy: {accuracies / 6 * 100}%")
plt.show()
sim.close()

# pytjaniki jeżeli za mała pewność
# odległość najlepszego rozwiązania od średniej lub kolejnego
# zbadać wsparcie dla opencla przy nengo
