import cv2
import os
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_faces(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    filename = {}
    extracted_faces = []
    # Loop over all detected faces
    for i, (x, y, w, h) in enumerate(faces):
        # Extract the detected face from the input image
        face = img[y:y+h, x:x+w]

        # Create a new image to hold the extracted face
        face_img = cv2.resize(face, (128, 128))

        # Save the extracted face as a PNG image
        filename[f'extracted_face_{i}.jpg'] = face_img
        # cv2.imwrite(filename, face_img)
        # Append the extracted face to a list
        extracted_faces.append(face_img)

