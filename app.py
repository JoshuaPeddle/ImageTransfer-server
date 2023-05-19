from flask import Flask , request, send_file
from flask_cors import CORS
from PIL import Image
from models.models import upscale, monet, gogh, picasso, dali
from random_image import generate_random_image_url

app = Flask(__name__)
CORS(app)

@app.route("/gogh", methods=['POST'])
def _gogh():
    return send_file(gogh(request_to_image(request)), mimetype='image/jpg')

@app.route("/monet", methods=['POST'])
def _monet():
    return send_file(monet(request_to_image(request)), mimetype='image/jpg')

@app.route("/picasso", methods=['POST'])
def _picasso():
    return send_file(picasso(request_to_image(request)), mimetype='image/jpg')

@app.route("/dali", methods=['POST'])
def _dali():
    return send_file(dali(request_to_image(request)), mimetype='image/jpg')

@app.route("/upscale", methods=['POST'])
def _upscale():
    return send_file(upscale(request_to_image(request)), mimetype='image/jpg')

'''
This route should return a url to a random images hosted from the github repo that that project is hosted from
'''
@app.route("/random", methods=['GET'])
def _random():
    url : str= generate_random_image_url() 
    return {'url': url}


@app.route("/", methods=['GET'])
def hello():
    return "Hello World"

'''
TODO: Should generalize the generate function to take in a model name and generate the image
'''
@app.route("/generate/<model>", methods=['POST'])
def _generate(model):
    pass


def request_to_image(request):
    image = request.files.get('image', False)
    print(image)
    if image:
        return Image.open(image)
    else:return Image.open(request.files['image'])