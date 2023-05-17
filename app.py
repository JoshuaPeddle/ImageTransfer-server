from flask import Flask , request, send_file
from flask_cors import CORS
from PIL import Image
from models.models import upscale, monet, gogh

app = Flask(__name__)
CORS(app)

@app.route("/gogh", methods=['POST'])
def _gogh():
    return send_file(gogh(request_to_image(request)), mimetype='image/jpg')

@app.route("/monet", methods=['POST'])
def _monet():
    return send_file(monet(request_to_image(request)), mimetype='image/jpg')

@app.route("/upscale", methods=['POST'])
def _upscale():
    return send_file(upscale(request_to_image(request)), mimetype='image/jpg')

@app.route("/generate/<model>", methods=['POST'])
def _generate(model):
    pass



@app.route("/", methods=['GET'])
def hello():
    return "Hello World"

def request_to_image(request):
    image = request.files.get('image', False)
    print(image)
    if image:
        return Image.open(image)
    else:return Image.open(request.files['image'])