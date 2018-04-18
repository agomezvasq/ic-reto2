import os
import cv2
import matplotlib
from sklearn.cluster import KMeans
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

PLOT = True
OVERWRITE = False

PATH = 'test/IMG_20180417_162819.jpg'

img = cv2.imread('test/IMG_20180417_162819.jpg')

#window = cv2.namedWindow('window', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('window', 1280, 720)

img = cv2.resize(img, (1280, int(1280 / (img.shape[1] / img.shape[0]))))

img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img_lab[:, :, 0] = 127

img_nol = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

#cv2.imshow('window', img)
#cv2.waitKey(0)

size = img.shape[0] * img.shape[1]
idx = np.random.randint(0, size, 100000)
sample = img_lab.reshape(size, 3)[idx, 1:3]

n_clusters = 2

kmeans = None

all_contours = []
all_contours_areas = []

colors = []

masks = []

outliers = []

while len(outliers) == 0:
    _kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(sample)

    pred = _kmeans.predict(img_lab[:, :, 1:3].reshape(size, 2))

    i = 0
    j = 0
    window_width = 300
    for window in range(n_clusters):
        cv2.namedWindow(str(window), cv2.WINDOW_NORMAL)
        cv2.resizeWindow(str(window), window_width, int(1.5 * window_width))
        cv2.moveWindow(str(window), i * (window_width + 100), int(j * (1.5 * window_width + 100)))
        i += 1
        if i > 4:
            i = 0
            j += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    _all_contours = []
    _all_contours_areas = []

    #colors = np.array([np.hstack((np.ones((n_clusters, 1)) * 127, np.array(_kmeans.cluster_centers_)))]).astype('uint8')
    #colors = cv2.cvtColor(colors, cv2.COLOR_LAB2BGR)

    _colors = []

    _masks = []

    for i in range(n_clusters):
        mask = (pred == i).reshape(img.shape[0], img.shape[1])
        cluster_img = np.zeros((img.shape[0], img.shape[1]))
        cluster_img[mask] = 255
        cluster_img = cluster_img.astype('uint8')
        cluster_img = cv2.morphologyEx(cluster_img, cv2.MORPH_CLOSE, kernel, iterations=5)
        cluster_img = cv2.morphologyEx(cluster_img, cv2.MORPH_OPEN, kernel, iterations=10)
        #cluster_img = cv2.merge((cluster_img, cluster_img, cluster_img))

        im, contours, hierarchy = cv2.findContours(cluster_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_masked = cv2.bitwise_and(img, img, mask=cluster_img)
        #cv2.drawContours(img_masked, contours, -1, (0, 0, 255), thickness=9)

        avg_color = cv2.mean(img, cluster_img)[0:3]
        #img_simple = np.zeros(img.shape, np.uint8)
        #img_simple[cluster_img] = avg_color
        _masks.append(cluster_img)

        _all_contours.append(contours)
        _all_contours_areas.append([cv2.contourArea(contour) for contour in contours])

        _colors.append(avg_color)

        cv2.imshow(str(i), img_masked)

    print('areas: ' + str(_all_contours_areas))
    arr = np.array([item for lst in sorted(_all_contours_areas, key=lambda x: sum(x), reverse=True)[1:] for item in lst])
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    print('mean: ' + str(mean) + ', std: ' + str(std))
    outliers = [x for x in arr if x < mean - 3.5 * std or x > mean + 3.5 * std]

    print('outliers: ' + str(outliers))

    if len(outliers) == 0:
        kmeans = _kmeans

        all_contours = _all_contours
        all_contours_areas = _all_contours_areas

        colors = _colors

        masks = _masks

    n_clusters += 1

    #cv2.waitKey(0)

img_contours = img.copy()
image_simple = np.ones(img.shape, np.uint8) * 255
contours = all_contours[1:]
colors = colors[1:]
masks = masks[1:]
for i in range(len(contours)):
    cv2.drawContours(img_contours, contours[i], -1, colors[i], thickness=9)

    color_img = np.zeros(img.shape, np.uint8)
    color_img[:, :] = np.array(list(colors[i]), np.uint8)
    color_img = cv2.bitwise_and(color_img, color_img, mask=masks[i])
    mask_inv = cv2.bitwise_not(masks[i])
    image_simple = cv2.bitwise_and(image_simple, image_simple, mask=mask_inv)
    image_simple = cv2.add(image_simple, color_img)

    cv2.drawContours(image_simple, contours[i], -1, (0, 0, 0), thickness=6)
    for contour in contours[i]:
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.putText(image_simple, str(i + 1), (cx - 30, cy + 30), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6, cv2.LINE_AA)

cv2.namedWindow('window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('window', 1280, 720)
cv2.imshow('window', np.hstack((img, image_simple)))
cv2.waitKey(0)


if PLOT:
    x = sample[:, 0]
    y = sample[:, 1]
    c = img_nol.reshape(size, 3)[idx, :] / 255.0

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y, s=8, c=c)

    centroids_x = kmeans.cluster_centers_[:, 0]
    centroids_y = kmeans.cluster_centers_[:, 1]

    ax.scatter(centroids_x, centroids_y, marker='+', c='red')

    plt.show()