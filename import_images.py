import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops


def load_images(img_label="BG_0014.tif"):
    img_path = "../segmentacja_output_przyciete/"
    files_to_load = []
    for img_name in os.listdir(img_path):
        i = img_name.rfind("_")
        label = img_name[:i]
        if label == img_label:
            index = int(img_name[i + 1 : -4])
            files_to_load.append((img_name, index))
    files_to_load.sort(key=lambda x: x[1])
    print(files_to_load)

    n_files = len(files_to_load)
    s = 28 * 28
    images = np.zeros((n_files, s))

    idx = 0
    for file, _ in files_to_load:
        I = cv2.imread(img_path + file)
        bgr = mnistify_image(I)
        images[idx, :] = bgr.flatten()
        idx += 1

    n_steps = 30
    test_images = np.tile(images[:, None, :], (1, n_steps, 1))
    return test_images


def mnistify_image(I):
    I = 255 - cv2.cvtColor(I, cv2.COLOR_BGR2GRAY).astype(np.float32)
    scale_factor = 20 / max(I.shape)
    I_resized = cv2.resize(I, (0, 0), fx=scale_factor, fy=scale_factor)
    _, I_bin = cv2.threshold(I_resized, 50, 255, cv2.THRESH_BINARY)

    M = cv2.moments(I_bin, 1)
    M00 = M["m00"]

    x, y = I_resized.shape
    x_start = int((28 - x) // 2 + 10 - M["m01"] / M00)
    x_end = x_start + x

    y_start = int((28 - y) // 2 + 10 - M["m10"] / M00)
    y_end = y_start + y

    bgr = np.zeros((28, 28))
    bgr[x_start:x_end, y_start:y_end] = I_resized
    return bgr


if __name__ == "__main__":
    a = load_images()
    for i in range(10):
        plt.figure()
        plt.imshow(a[i, 0].reshape((28, 28)))
plt.show()
