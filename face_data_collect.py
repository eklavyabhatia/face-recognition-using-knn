# Write a Python Script that captures images from your webcam vide stream
# Extract all faces from the image frame(using harcascade)
# Store the information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box
# 3. Flatten the largest face image and save in numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

# Init Camera
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
colected_images = 0

person_name = input()
face_data = []

while True:

    ret, frame = cap.read()

    if ret == False:
        continue

    # gray_frame can be used to save memory
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    # Store Every 10th face

    if skip % 10 == 0:
        if len(faces) > 0:
            lf = faces[0]
            x = lf[0]
            y = lf[1]
            w = lf[2]
            h = lf[3]

            offset = 10

            crop_face = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            cv2.imshow('Cropped Face', crop_face)

            crop_face = cv2.resize(crop_face, (100, 100))
            crop_face_flatten = np.reshape(crop_face, (-1,))
            face_data.append(crop_face_flatten)
            colected_images = colected_images+1
    skip = skip+1

    if colected_images == 20:
        break
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

face_data = np.array(face_data)
np.save('./data/'+person_name+'.npy', face_data,)

cap.release()
cv2.destroyAllWindows()
