import os
import face_recognition
import shutil
import cv2
import uuid



users_dir = "flaskr/users"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
namespace = uuid.UUID('00000000-0000-0000-0000-000000000000')  # Fixed namespace identifier

def extract_faces(img_path):
    # Load the input image
    img = cv2.imread(img_path)

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
        # filename[f'extracted_face_{i}.jpg'] = face_img
        # cv2.imwrite(filename, face_img)
        # Append the extracted face to a list
        extracted_faces.append(face_img)
    return extracted_faces



def recognize_faces(img_path, img_name, token):
    folder_name = str(uuid.uuid5(namespace, token))
    folder_path = users_dir + "/id_" + folder_name
    if not os.path.exists(folder_path):
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
    extracted_faces = extract_faces(img_path)
    # face_locations = face_recognition.face_locations(img, model="cnn")
    if(len(extracted_faces) == 0):
        return "I can see no one here"
    else:
        # img = face_recognition.load_image_file(img_path)
        img_encoding = face_recognition.face_encodings(extracted_faces[0])[0]
        # Iterate over each file in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # load familiar faces images and compare them to the given img
            if os.path.isfile(file_path):
                fam_face_img = face_recognition.load_image_file(file_path)
                fam_face_encoding = face_recognition.face_encodings(fam_face_img)[0]
                results = face_recognition.compare_faces([img_encoding], fam_face_encoding, tolerance=0.5)
                if results[0] == True:
                    return filename.split(".")[0]
        new_path = os.path.join(folder_path, img_name)
        # Copy the image file to the user folder
        shutil.copy2(img_path, new_path)
        # rename img to temp
        new_name_path = os.path.join(folder_path, 'temp.jpg')
        if os.path.exists(new_name_path):
            os.remove(new_name_path)
        os.rename(new_path, new_name_path)
        return "New face has been detected"
        

def create_new_user(token):
    folder_name = str(uuid.uuid5(namespace, token))
    folder_path = users_dir+"/id_"+folder_name
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def add_to_friends_list(name, token):
    folder_name = str(uuid.uuid5(namespace, token))
    folder_path = users_dir + "/id_" + folder_name
    temp_path =  os.path.join(folder_path, 'temp.jpg')
    new_name_path = os.path.join(folder_path, name+'.jpg')
    os.rename(temp_path, new_name_path)
    return name+" has been added to your friends list"



