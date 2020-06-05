# Recognizing faces using some classification algorithms - knn, svm, logistic

# 1. Read a video stream using opencv
# 2. Extract faces out of it (Testing)
# 3. Load the training data of(numpy arrays of all the persons)
# x - valuse are stored in the numpy arrays
# y - values we need to assign to each person
# 4. use knn to find predictions of face
# 5. map the predicted id to the name of the user(from the dictionary)
# 6. Display the predictions on the screen - bounding box and name

import cv2
import numpy as np
import os

# knn


def dist(x1, x2):
    return np.sqrt(sum((x1-x2)**2))


def knn(X, Y, query_point, k):

    vals = []

    total_points = X.shape[0]

    for i in range(total_points):
        dist_from_ith_point = dist(X[i], query_point)
        t = (dist_from_ith_point, Y[i])
        vals.append(t)

    vals = sorted(vals)

    vals = vals[:k]

    classes = []
    for i in range(k):
        classes.append(vals[i][1])

    classes = np.array(classes)

    res = np.unique(classes, return_counts=True)
    max_freq_index = np.argmax(res[1])
    return res[0][max_freq_index]


cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Data preprocessing
class_id = 0
names = {}
X = list()
Y = list()

for fx in os.listdir('./data'):
    arr = np.load('./data/'+fx)

    for i in range(arr.shape[0]):
        X.append(arr[i])
        Y.append(class_id)

    names[class_id] = fx.split('.')[0]
    class_id = class_id+1

X = np.array(X)
Y = np.array(Y)

# Testing

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    faces = face_classifier.detectMultiScale(frame, 1.3, 5)

    for cf in faces:
        x = cf[0]
        y = cf[1]
        w = cf[2]
        h = cf[3]
        offset = 10
        crop_image = frame[y-offset:y+offset+h, x-offset:x+offset+w]
        crop_image = cv2.resize(crop_image, (100, 100))
        crop_image_flatten = np.reshape(crop_image, (-1,))

        pp = knn(X, Y, crop_image_flatten, 5)

        cv2.putText(frame, names[pp], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Frame', frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
