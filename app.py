from flask import Flask , request, send_file, Request
from flask import current_app, g as app_ctx
from flask_cors import CORS
from PIL import Image
from models.models import generate, load_model, get_styles
from random_image import generate_random_image_url
import functools
import time

app = Flask(__name__)
CORS(app)

@app.before_request
def logging_before():
    # Store the start time for the request
    app_ctx.start_time = time.perf_counter()


@app.after_request
def logging_after(response):
    # Get total time in milliseconds
    total_time = time.perf_counter() - app_ctx.start_time
    time_in_ms = int(total_time * 1000)
    # Log the time taken for the endpoint 
    print(f'{time_in_ms} ms {request.method} {request.path}')
    return response

'''
This route should return a url to a random images hosted from the github repo that that project is hosted from
'''
@app.route("/random", methods=['GET'])
def _random():
    url : str= generate_random_image_url() 
    return {'url': url}

'''
TODO: Should generalize the generate function to take in a model name and generate the image
'''
@app.route("/generate/<model>", methods=['POST'])
def _generate(model):
    byte_arr = generate(request_to_image(request), model)
    return send_file(byte_arr, mimetype='image/jpg')

'''
TODO: Should generalize the generate function to take in a model name and generate the image
'''
@app.route("/generate/<model>/<variant>", methods=['POST'])
def _generate_variant(model, variant):
    image , uuid = request_to_image(request)
    byte_arr = generate(image, style=model, variant=int(variant), uuid=uuid)
    return send_file(byte_arr, mimetype='image/jpg')

@app.route("/styles", methods=['GET'])
def styles():
    return get_styles()

@app.route("/", methods=['GET'])
def hello():
    return "Hello World"

def request_to_image(request: Request):
    image = request.files.get('image', False)
    if not image:
        image  = request.files['image']
    uuid = image.filename
    uuid = uuid[:-4]
    return Image.open(image), uuid

# push context manually to app
with app.app_context():
    load_model()
