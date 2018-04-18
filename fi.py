import cv2
import matplotlib
from sklearn.cluster import KMeans
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

PLOT = False

#window = cv2.namedWindow('window', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('window', 1280, 720)

img = cv2.imread('test/IMG_20180417_162819.jpg')

img = cv2.resize(img, (1280, int(1280 / (img.shape[1] / img.shape[0]))))

img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img_lab[:, :, 0] = 127

img_nol = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

#cv2.imshow('window', img)
#cv2.waitKey(0)

size = img.shape[0] * img.shape[1]
idx = np.random.randint(0, size, 100000)
sample = img_lab.reshape(size, 3)[idx, 1:3]

N_CLUSTERS = 6

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(sample)

pred = kmeans.predict(img_lab[:, :, 1:3].reshape(size, 2))

i = 0
j = 0
size = 500
for window in range(N_CLUSTERS):
    cv2.namedWindow(str(window), cv2.WINDOW_NORMAL)
    cv2.resizeWindow(str(window), size, size)
    cv2.moveWindow(str(window), i * (size + 100), j * (size + 100))
    i += 1
    if i > 2:
        i = 0
        j += 1

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

colors = kmeans.cluster_centers_

all_contours = []

for i in range(N_CLUSTERS):
    mask = (pred == i).reshape(img.shape[0], img.shape[1])
    cluster_img = np.zeros((img.shape[0], img.shape[1]))
    cluster_img[mask] = 255
    cluster_img = cluster_img.astype('uint8')
    cluster_img = cv2.morphologyEx(cluster_img, cv2.MORPH_CLOSE, kernel, iterations=5)
    cluster_img = cv2.morphologyEx(cluster_img, cv2.MORPH_OPEN, kernel, iterations=10)
    #cluster_img = cv2.merge((cluster_img, cluster_img, cluster_img))

    im, contours, hierarchy = cv2.findContours(cluster_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_masked = cv2.bitwise_and(img, img, mask=cluster_img)
    cv2.drawContours(img_masked, contours, -1, (0, 0, 255), thickness=9)

    cv2.imshow(str(i), img_masked)
cv2.waitKey(0)


if PLOT:
    x = sample[:, 0]
    y = sample[:, 1]
    c = img.reshape(size, 3)[idx, :] / 255.0

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y, s=8, c=c)

    centroids_x = kmeans.cluster_centers_[:, 0]
    centroids_y = kmeans.cluster_centers_[:, 1]

    ax.scatter(centroids_x, centroids_y, marker='+', c='red')

    plt.show()