import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage import io
from skimage.measure import label, regionprops


def segmentoutletters(name):
    im = np.asarray(Image.open("../Dane/" + name))
    im_orig = im.copy()
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    _, im = cv.threshold(~im, 173, 255, cv.THRESH_BINARY)
    # im = cv.morphologyEx(im, cv.MORPH_CLOSE, (3, 3))
    # im = cv.medianBlur(im, 5)
    # im = cv.morphologyEx(im, cv.MORPH_CLOSE, (3, 3))

    img = label(im)
    props = regionprops(img)
    # io.imshow(img)

    bboxes = np.array([i.bbox for i in props])
    bbox_areas, bbox_verticality = np.array(
        [((c - a) * (d - b), (c - a) / (d - b)) for a, b, c, d in bboxes]
    ).T

    low = 5000
    high = 15000
    vert = 1
    bboxes_filtered = bboxes[
        np.logical_and(
            np.logical_and(bbox_areas < high, bbox_areas > low), bbox_verticality > vert
        )
    ]
    # display
    colored_image = bbox_areas[img - 1]
    c_i2 = bbox_verticality[img - 1]
    colored_image = np.logical_and(
        np.logical_and(colored_image < high, colored_image > low), c_i2 > vert
    )
    fig = plt.figure()
    fig.canvas.manager.set_window_title(name)
    plt.imshow(colored_image * 255 / colored_image.max())

    # fig, ax = plt.subplots(2, 5)
    # fig.canvas.manager.set_window_title(name + "_segment")
    # for i in range(10):
    #     plt.subplot(4, 3, i + 1)
    #     minr, minc, maxr, maxc = bboxes_filtered[i]
    #     wind = im_orig[minr:maxr, minc:maxc, :]
    #     wind_save = Image.fromarray(wind)
    #     wind_save.save(f"../segmentacja_output/{name[:-4]}_{i}.jpg")
    #     plt.imshow(cv.resize(wind, (32, 32)))
    i = 0
    for minr, minc, maxr, maxc in bboxes_filtered:
        wind = im_orig[minr:maxr, minc:maxc, :]
        wind_save = Image.fromarray(wind)
        wind_save.save(f"../segmentacja_output/{name[:-4]}_{i}.jpg")
        i += 1


from datetime import datetime

names = os.listdir("../Dane")
# segmentoutletters(names[0])
for name in names:
    t1 = datetime.now()
    segmentoutletters(name)
    t2 = datetime.now()
    print((t2 - t1))
plt.show()
