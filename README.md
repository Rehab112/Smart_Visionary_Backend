# Smart Visionary Backend

#### Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Workflow](#workflow)
4. [Prerequisites](#prerequisites)
5. [API Endpoints](#api-endpoints)
6. [Flutter App](#flutter-app)

#### Overview
The backend for the Smart Visionary application, designed to assist the visually impaired and blind. This repository contains the API server built using Flask and deep learning models deployed on it.

#### Features
1. **Face Recognition** - Recognizes familiar faces.
2. **Image Captioning** - Describes an image in a sentence.
3. **Object Detection** -  Detect specified objects in an image and estimate their depth.
4. **OCR** - Optical Character Recognition for reading text in images.
5. **Money Recognition** - Trained to recognize Egyptian currency using YOLO v8.
6. **Translation** - Translates input text into the Arabic language.

#### Workflow
Once the user chooses a model or feature in the app and double-taps to capture an image, the image is sent to the backend through the specified API for the chosen feature.
The backend processes the image using the selected model and sends back the result in the form of text within a JSON file.
The frontend application (Flutter app) then converts this text into audio and plays it for the user.

#### Prerequisites
- Python 3.9
- Flask
- CUDA (for YOLO v8)
- Other dependencies in `requirements.txt`

#### API Endpoints
- **Money Recognition** - `POST /money`
- **Image Captioning** - `POST /caption`
- **Face Recognition** 
  - Recognizing Faces - `POST /face`
  - Add Friend to Recognition List - `POST /face/add_friend`
  - Create New User for Face Recognition - `POST /new_user`
- **Object Detection** - `POST /detect`
- **OCR** - `POST /ocr`
- **Text Translation** - `POST /trans`
  
#### Flutter App
- [Frontend Repository (Flutter App)](https://github.com/Rehab112/Smart_Visionary_App)

---
