#PROGRAM TO FACE DETECTION USING OPEN CV IN A PICTURE

import cv2

# Load the cascade

fcas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Read the input image

pic = cv2.imread("test.jpg")

# Convert into grayscale

gray_img = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

# Detect faces

faces = fcas.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)

# Draw rectangle around the faces

for (x, y, w, h) in faces:
    cv2.rectangle(pic, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output

cv2.imshow('img', pic)
cv2.waitKey()
