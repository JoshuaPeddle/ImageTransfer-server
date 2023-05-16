

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


from models.Monet import MonetGenerator
from models.Upscale import UpscaleGenerator


upscale_generator = UpscaleGenerator()
monet_generator = MonetGenerator()



def load_models():
    upscale_generator.load_model()
    monet_generator.load_model()

def upscale(image):
    return upscale_generator.generate(image)


def monet(image):
    return monet_generator.generate(image)





load_models()