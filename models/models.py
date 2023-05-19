from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from models.Monet import MonetGenerator
from models.Upscale import UpscaleGenerator
from models.Gogh import GoghGenerator
from models.Picasso import PicassoGenerator
from models.Dali import DaliGenerator

upscale_generator = UpscaleGenerator()
monet_generator = MonetGenerator()
gogh_generator = GoghGenerator()
picasso_generator = PicassoGenerator()
dali_generator = DaliGenerator()

def load_models():
    upscale_generator.load_model()
    monet_generator.load_model()
    gogh_generator.load_model()
    picasso_generator.load_model()
    dali_generator.load_model()

def upscale(image):
    return upscale_generator.generate(image)

def monet(image):
    return monet_generator.generate(image)

def gogh(image):
    return gogh_generator.generate(image)

def picasso(image):
    return picasso_generator.generate(image)

def dali(image):
    return dali_generator.generate(image)

load_models()