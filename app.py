from flask import Flask , request, send_file
from PIL import Image
from models.models import _upscale, _monet, request_to_image
from numpy import asarray

app = Flask(__name__)

@app.route("/monet", methods=['POST'])
def monet():
    return send_file(_monet(request_to_image(request) ,True), mimetype='image/jpg')

@app.route("/upscale", methods=['POST'])
def upscale():
    print(request.files)
    return send_file(_upscale(request_to_image(request)), mimetype='image/jpg')