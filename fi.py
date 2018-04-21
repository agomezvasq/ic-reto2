#Andres Gomez

import os
import cv2
import matplotlib
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

PLOT = True

img = cv2.imread('test/IMG_20180417_162632.jpg')

cv2.namedWindow('window', cv2.WINDOW_NORMAL)
cv2.resizeWindow('window', 1280, 720)

img = cv2.resize(img, (1280, int(1280 / (img.shape[1] / img.shape[0]))))

img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img_lab[:, :, 0] = 0

img_nol = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

#cv2.imshow('window', img)
#cv2.waitKey(0)

size = img.shape[0] * img.shape[1]
idx = np.random.randint(0, size, 100000)
sample = img_lab.reshape(size, 3)[idx, 1:3]
cv2.imshow('window', img_nol)
cv2.imwrite('img_nol.png', img_nol)
cv2.waitKey(0)

n_clusters = 2

kmeans = None

all_contours = []
all_contours_areas = []

colors = []

masks = []

outliers = []

mean = 0
std = 0

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
        save_cluster_img = cv2.merge((cluster_img, cluster_img, cluster_img))
        #cv2.imwrite('cluster' + str(i) + '.png', save_cluster_img)

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
    idxs = [x[0] for x in sorted(enumerate(_all_contours_areas), key=lambda x: sum(x[1]), reverse=True)]
    _all_contours_areas = [_all_contours_areas[i] for i in idxs]
    _all_contours = [_all_contours[i] for i in idxs]
    _colors = [_colors[i] for i in idxs]
    _masks = [_masks[i] for i in idxs]
    arr = np.array([item for lst in _all_contours_areas[1:] for item in lst])
    if mean == 0:
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
    print('mean: ' + str(mean) + ', std: ' + str(std))
    outliers = [x for x in arr if x < mean - 3.5 * std or x > mean + 3.5 * std]
    if len(arr) == 1:
        std = 1000
    if len(arr) == 0:
        mean = 0
        std = 0

    print('outliers: ' + str(outliers))

    if len(outliers) == 0:
        kmeans = _kmeans

        all_contours = _all_contours
        all_contours_areas = _all_contours_areas

        colors = _colors

        masks = _masks

        #mean = np.mean(arr, axis=0)
        #std = np.std(arr, axis=0)

    n_clusters += 1

    #cv2.waitKey(0)

img_contours = img.copy()
image_simple = np.ones(img.shape, np.uint8) * 255
contours = all_contours[1:]
colors = colors[1:]
masks = masks[1:]
font_size = math.sqrt(mean / math.pi) / 18
objects = {}
for i in range(len(contours)):
    cv2.drawContours(img_contours, contours[i], -1, colors[i], thickness=9)

    avg_color = cv2.mean(img, masks[i])[0:3]
    color_img = np.zeros(img.shape, np.uint8)
    color_img[:, :] = np.array(list(avg_color), np.uint8)
    color_img = cv2.bitwise_and(color_img, color_img, mask=masks[i])
    mask_inv = cv2.bitwise_not(masks[i])
    image_simple = cv2.bitwise_and(image_simple, image_simple, mask=mask_inv)
    image_simple = cv2.add(image_simple, color_img)

    cv2.drawContours(image_simple, contours[i], -1, (0, 0, 0), thickness=6)
    for j in range(len(contours[i])):
        M = cv2.moments(contours[i][j])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.putText(image_simple, str(j + 1), (cx - int(font_size * 10), cy + int(font_size * 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 7, cv2.LINE_AA)
    objects.update({i: len(contours[i])})
print(objects)


show_img = np.hstack((img, image_simple))
show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
fig = plt.figure()
ax = plt.subplot(111)
ax.imshow(show_img)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
circles = []
for i in range(len(contours)):
    color = colors[i]

    line = Line2D(range(1),
                  range(1),
                  color="white",
                  marker='o',
                  markerfacecolor=list(np.array([color[2], color[1], color[0]]) / 255),
                  markeredgecolor='black',
                  markersize=12)
    circles.append(line)
ax.axis('off')
ax.legend(tuple(circles),
          tuple([str(objects[i]) for i in range(len(contours))]),
          loc='center left',
          bbox_to_anchor=(1, 0.5))
plt.show()
#cv2.namedWindow('window', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('window', 1280, 720)
#cv2.imshow('window', np.hstack((img, image_simple)))
#cv2.waitKey(0)




if PLOT:
    x = sample[:, 0]
    y = sample[:, 1]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    c = img_rgb.reshape(size, 3)[idx, :] / 255.0

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y, s=8, c=c)

    centroids_x = kmeans.cluster_centers_[:, 0]
    centroids_y = kmeans.cluster_centers_[:, 1]

    ax.scatter(centroids_x, centroids_y, marker='+', c='red')

    plt.show()
