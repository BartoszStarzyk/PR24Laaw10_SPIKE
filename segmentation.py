import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_object
from datetime import datetime
from import_images import mnistify_image

S_THRESH_L = 30
S_THRESH_H = 150
B_THRESH = 100
S_CHAR_THRESH = 1.1

UNDERSCORE_ID = -1
SLASH_ID = -2
QUESTIONMARK_ID = -3


def segmentoutletters(name, save=False, detect_special_chars=False):
    img = np.asarray(Image.open("Dane_przyciete/" + name))
    img_orig = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(~img, B_THRESH, 255, cv.THRESH_BINARY)
    img = cv.medianBlur(img, 5)

    labels = label(img)
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
    flat_img_size = 28 * 28
    images = np.zeros((n_images, flat_img_size))
    special_char = np.zeros((n_images,))
    for minr, minc, maxr, maxc in b_boxes_filt:
        region_slice = np.s_[minr:maxr, minc:maxc]
        window = img_orig[region_slice]
        if detect_special_chars:
            char_image = img[region_slice]
            conv_area = np.count_nonzero(convex_hull_object(char_image))
            char_area = np.count_nonzero(char_image)
            if conv_area / char_area <= S_CHAR_THRESH:
                if maxr - minr >= maxc - minc:
                    special_char[i] = SLASH_ID
                else:
                    special_char[i] = UNDERSCORE_ID
        if save:
            wind_save = Image.fromarray(window)
            wind_save.save(f"segmentacja_output_przyciete/{name[:-4]}_{i}.jpg")
        else:
            window = mnistify_image(window)
            images[i, :] = window.flatten()
        i += 1

    time_steps = 30
    images = np.tile(images[:, None, :], (1, time_steps, 1)).astype(np.uint8)
    if detect_special_chars:
        return images, img_orig, special_char
    return images, img_orig


if __name__ == "__main__":
    names = os.listdir("Dane_przyciete")
    temp_counter = 0
    for name in names:
        t1 = datetime.now()
        _, _, schars = segmentoutletters(name, False, True)
        print(name, schars)
        t2 = datetime.now()
        print((t2 - t1))
        temp_counter += 1
    plt.show()
