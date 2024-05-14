import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.measure import label, regionprops
from datetime import datetime
from import_images import mnistify_image

S_THRESH_L = 30
S_THRESH_H = 150
B_THRESH = 100


def segmentoutletters(name, save=False):
    im = np.asarray(Image.open("../Dane_przyciete/" + name))
    im_orig = im.copy()
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    _, im = cv.threshold(~im, B_THRESH, 255, cv.THRESH_BINARY)
    im = cv.medianBlur(im, 5)

    labels = label(im)
    props = regionprops(labels)

    b_boxes = np.array([i.bbox for i in props])
    height, width = np.array(
        [((maxr - minr), (maxc - minc)) for minr, minc, maxr, maxc in b_boxes]
    ).T

    b_boxes_filt = b_boxes[
        np.logical_or(
            np.logical_and(height < S_THRESH_H, height > S_THRESH_L),
            np.logical_and(width < S_THRESH_H, width > S_THRESH_L),
        )
    ]
    i = 0

    b_boxes_filt = sorted(list(b_boxes_filt), key=lambda x: x[1])
    n_images = len(b_boxes_filt)
    s = 28 * 28
    images = np.zeros((n_images, s))

    for minr, minc, maxr, maxc in b_boxes_filt:
        wind = im_orig[minr:maxr, minc:maxc, :]
        if save:
            wind_save = Image.fromarray(wind)
            wind_save.save(f"../segmentacja_output_przyciete/{name[:-4]}_{i}.jpg")
        else:
            wind = mnistify_image(wind)
            images[i, :] = wind.flatten()
        i += 1

    n_steps = 30
    images = np.tile(images[:, None, :], (1, n_steps, 1))
    return images, im_orig


names = os.listdir("../Dane")
for name in names:
    t1 = datetime.now()
    segmentoutletters(name, save=True)
    t2 = datetime.now()
    print((t2 - t1))
plt.show()
