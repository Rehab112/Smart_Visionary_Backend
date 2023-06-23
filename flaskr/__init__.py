from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from PIL import Image
import textract
from flaskr import functions as fun
from .models import object_detection_model as det
from .models import image_captioning_model as cap
from .models import money_recognition_model as rec
import ultralytics

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


    # a simple page that says hello
    @app.route('/money', methods=['POST'])
    def recognition():
         # check if the post request has the file part
        file = fun.check_image_file()
        if isinstance(file, tuple):
            return file  # Error response
        
        # Save the image to the server
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_name = filename.split(".")[0]
        file.save(file_path)

        # Perform OCR text extraction on the image
        text = rec.predict(file_path, img_name)
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
    

    @app.route('/detect', methods=['POST'])
    def object_detection():
        # check if the post request has the file part
        file = fun.check_image_file()
        if isinstance(file, tuple):
          return file  # Error response
        
        image = Image.open(file)
        txt = request.form.get('text')
        
        objects = []
        if len(txt.split()) == 1:
            objects = [txt]  # convert input to list with one element
        else:
            objects = txt.split()

        print(objects)
        # Perform image captioning on the image
        print("processing")
        output = det.detect_objects(objects, image)
            
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
        text = textract.process(file_path, language='eng').decode('utf-8')
        print(text)

        # Send the text to the Flutter app
        response_data = {'text': text}
        # Delete the saved photo from the server
        os.remove(file_path)
        
        return jsonify(response_data), 200


    @app.route('/')
    def hello():
        return 'Hello from vision App!'
    
    

    return app
