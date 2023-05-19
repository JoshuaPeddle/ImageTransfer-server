from flask import Flask , request, send_file, Request
from flask_cors import CORS
from PIL import Image
from models.models import generate, is_loaded, load_model
from random_image import generate_random_image_url

app = Flask(__name__)
CORS(app)


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
    if not is_loaded(model):
        load_model(model)
    return send_file(generate(request_to_image(request), model), mimetype='image/jpg')

@app.route("/", methods=['GET'])
def hello():
    return "Hello World"


def request_to_image(request: Request):
    image = request.files.get('image', False)
    print(image)
    if image:
        return Image.open(image)
    else:return Image.open(request.files['image'])

# # push context manually to app
# with app.app_context():
#     load_models()
