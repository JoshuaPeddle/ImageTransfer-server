from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from models.Monet import MonetGenerator
from models.Gogh import GoghGenerator
from models.Picasso import PicassoGenerator
from models.Dali import DaliGenerator


generators = {
    "monet": MonetGenerator(),
    "gogh": GoghGenerator(),
    "picasso": PicassoGenerator(),
    "dali": DaliGenerator()
}

def load_models():
    for generator in generators.values():
        generator.load_model()


def generate(image, style):
    return generators[style].generate(image)

load_models()