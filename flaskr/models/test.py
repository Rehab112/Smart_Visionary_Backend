import pytesseract
from PIL import Image
import cv2
from gtts import gTTS
import os
import numpy as np
from matplotlib import pyplot as plt


# # Path to Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# Path to input image file
os.chdir('d:/ASU/GP/flask_backend/flaskr/ai_models')
image_path = 'download.png'

# # Read the image file
# img = cv2.imread(image_path)
# # Convert image to grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Apply thresholding to segment the text
# #thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# print(cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# thresh = cv2.threshold(gray, 0, 255, 10)[1]
# print(np.min(thresh))
# print(np.max(thresh))

# # Apply dilation to make the text more visible
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# dilate = cv2.dilate(thresh, kernel, iterations=1)

# plt.imshow(gray,cmap='gray')
# plt.show()

# # Use Tesseract OCR to extract and recognize text
# text = pytesseract.image_to_string(gray)
# print(text)
# # Convert text to speech using gTTS
# tts = gTTS(text)
# tts.save("output_english.mp3")

# # Play the speech
# os.system("start output_english.mp3")

# from IPython.display import Audio

# # Load the saved mp3 file
# audio_file = 'output_english.mp3'

# # Play the audio file
# Audio(audio_file, autoplay=True)


"""## Testing with Arabic words using Kraken"""

import cv2
import pytesseract
import numpy as np

# read image and convert to grayscale
img = cv2.imread('/content/A.png')
img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

gray = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)

# Create a Laplacian filter
laplacian_filter = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])

# Apply the filter to the image
sharpened_img = cv2.filter2D(gray, -1, laplacian_filter)

# apply thresholding to binarize the image
thresh = cv2.threshold(sharpened_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# apply dilation to make text more prominent
kernel = np.ones((5,5),np.uint8)
dilate = cv2.dilate(thresh, kernel, iterations=1)

# # apply median blur to reduce noise
blur = cv2.medianBlur(sharpened_img, 3)

# apply image pre-processing techniques as required

# configure pytesseract to recognize Arabic text
config = r'--psm 3 --oem 3 -l ara'

# perform OCR on the pre-processed image
text_arabic = pytesseract.image_to_string(sharpened_img, config=config) # config=config

# print recognized text
print(text_arabic)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load image from file
# img = Image.open('1.jpg')

# Plot image
plt.imshow(sharpened_img, cmap = 'gray')
plt.show()

tts = gTTS(text_arabic, lang='ar')
tts.save("output_arabic.mp3")

from IPython.display import Audio

# Load the saved mp3 file
audio_file = 'output_arabic.mp3'

# Play the audio file
Audio(audio_file, autoplay=True)



img = cv2.imread('R.jpeg')

# Create a Laplacian filter
laplacian_filter = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])

# Apply the filter to the image
sharpened_img = cv2.filter2D(img, -1, laplacian_filter)

# Display the original and sharpened image
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].set_title('Original')
ax[1].imshow(sharpened_img)
ax[1].set_title('Sharpened')
plt.show()

