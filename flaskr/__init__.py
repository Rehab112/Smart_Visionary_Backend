from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from PIL import Image
from flaskr import functions as fun
from .models import object_detection_model as obj
from .models import image_captioning_model as cap
from .models import money_recognition_model as money
from .models import face_recognition_model as face
from .models import ocr_model as ocr
from .models import translation_model as trans

#RehabHosam.pythonanywhere.com.

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    CORS(app, resources={r"/*"})
    app.config['UPLOAD_FOLDER'] = 'uploads'
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['TIMEOUT'] = 600 # sets the timeout to 10 minutes
    app.config.from_mapping(
        SECRET_KEY='dev',
        # DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass


    
    
    @app.route('/money', methods=['POST'])
    def recognition():
         # check if the post request has the file part
        file = fun.check_image_file()
        if isinstance(file, tuple):
            return file  # Error response
        
        # Save the image to the server
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # img_name = filename.split(".")[0]
        file.save(file_path)

        # Perform OCR text extraction on the image
        text = money.predict(file_path)
        print(text)

        # Send the text to the Flutter app
        response_data = {'text': text}
        # Delete the saved photo from the server
        os.remove(file_path)
        
        return jsonify(response_data), 200
    
        
    @app.route('/caption', methods=['POST'])
    def image_captioning():
        # check if the post request has the file part
        file = fun.check_image_file()
        if isinstance(file, tuple):
          return file  # Error response
        image = Image.open(file)

        # Perform image captioning on the image
        print("processing")
        generated_caption = cap.generate_caption_from_image(image)
        print(generated_caption)

        # Send the text to the Flutter app
        response_data = {'text': generated_caption}
        return jsonify(response_data), 200
    


    @app.route('/face', methods=['POST'])
    def face_recognition():
        # check if the post request has the file part
        file = fun.check_image_file()
        if isinstance(file, tuple):
          return file  # Error response
        
        # Save the image to the server
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_name = filename.split(".")[0]
        file.save(file_path)        
        token = request.form.get('text')

        # start performing face recognition on the img
        print("processing")
        recognized_names = face.recognize_faces(file_path, img_name, token)
        # print(recognized_names)

        # Send the text to the Flutter app        
        response_data = {'text': recognized_names}
        os.remove(file_path)
        return jsonify(response_data), 200
    
    @app.route('/face/add_friend', methods=['POST'])
    def add_friend():
        req_data = request.form.get('text').split()
        return jsonify({'text': face.add_to_friends_list(req_data[0], req_data[1]) }), 200

    
    @app.route('/new_user', methods=['POST'])
    def create_user():
        token = request.form.get('text')
        face.create_new_user(token)
        # Send the text to the Flutter app        
        return jsonify({'text': "user created!"}), 200


    @app.route('/detect', methods=['POST'])
    def object_detection():
        # check if the post request has the file part
        file = fun.check_image_file()
        if isinstance(file, tuple):
          return file  # Error response
        
        image = Image.open(file)
        txt = request.form.get('text')
        
        objects = [txt]
        # if len(txt.split()) == 1:
        #     objects = [txt]  # convert input to list with one element
        # else:
        #     objects = txt.split()

        print(objects)
        # Perform image captioning on the image
        print("processing")
        output = obj.detect_objects(objects, image)
            
        print(output)

        # Send the text to the Flutter app
        response_data = {'text': output}
        return jsonify(response_data), 200
    
    @app.route('/ocr', methods=['POST'])
    def read_txt():
    # check if the post request has the file part
        file = fun.check_image_file()
        if isinstance(file, tuple):
            return file  # Error response
        
        # Save the image to the server
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform OCR text extraction on the image
        text = ocr.read_flipped(file_path)
        # text = ocr.read(file_path, 'eng')
        print(text)

        # Send the text to the Flutter app
        response_data = {'text': text}
        # Delete the saved photo from the server
        os.remove(file_path)
        
        return jsonify(response_data), 200
    
    @app.route('/trans', methods=['POST'])
    def translate_txt():
        txt = request.form.get('text')
        translated_txt = trans.translate(txt)
        # Send the text to the Flutter app        
        return jsonify({'text': translated_txt}), 200
    
    @app.route('/')
    def hello():
        return 'Hello from vision App!'
    
    

    return app
