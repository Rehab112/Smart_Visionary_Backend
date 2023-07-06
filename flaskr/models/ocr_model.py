import cv2
import pytesseract
import numpy as np
import textract



pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def read_flipped(img_path):
    return textract.process(img_path, language='eng').decode('utf-8')

def read(image_path, lang):
    
    img = cv2.imread(image_path)
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)
    laplacian_filter = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpened_img = cv2.filter2D(gray, -1, laplacian_filter) 
    if(lang == 'eng'):
           return pytesseract.image_to_string(sharpened_img) 

    # configure pytesseract to recognize Arabic text
    config = r'--psm 3 --oem 3 -l ara'
    text_arabic = pytesseract.image_to_string(sharpened_img, config=config) 
    return text_arabic

    