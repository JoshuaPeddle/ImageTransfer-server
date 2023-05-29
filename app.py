import time
from flask import Flask, request, send_file
from flask import g as app_ctx
from flask_cors import CORS
from models.models import generate, load_model, get_styles
from models.utils import request_to_image
from random_image import generate_random_image_urls


app = Flask(__name__)
CORS(app)


"""
This route should return a url to a random images hosted 
from the github repo that that project is hosted from
"""
@app.route("/random", methods=["GET"])
def _random():
    url: str = generate_random_image_urls(1)
    return {"urls": [url]}


"""
This route should return a json response with N urls to a random images hosted
from the github repo that that project is hosted from
"""
@app.route("/random/<n>", methods=["GET"])
def _nRandom(n):
    urls: str = generate_random_image_urls(n)
    return {"urls": urls}


"""
TODO: Should generalize the generate function to take
 in a model name and generate the image
"""
@app.route("/generate/<model>", methods=["POST"])
def _generate(model):
    image, uuid = request_to_image(request)
    byte_arr = generate(image, style=model, variant=None, uuid=uuid)
    return send_file(byte_arr, mimetype="image/jpg")


"""
TODO: Should generalize the generate function to take
 in a model name and generate the image
"""
@app.route("/generate/<model>/<variant>", methods=["POST"])
def _generate_variant(model, variant):
    image, uuid = request_to_image(request)
    byte_arr = generate(image, style=model, variant=int(variant), uuid=uuid)
    return send_file(byte_arr, mimetype="image/jpg")


@app.route("/styles", methods=["GET"])
def styles():
    return get_styles()


@app.route("/", methods=["GET"])
def hello():
    return "Hello World"

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
    print(f"{time_in_ms} ms {request.method} {request.path}")
    return response

# push context manually to app
with app.app_context():
    load_model()
