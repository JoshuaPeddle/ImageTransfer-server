from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


from models.Generator import Generator

models = {
    "monet": "MonetGenerator",
    "gogh": "GoghGenerator",
    "picasso": "PicassoGenerator",
    "dali": "DaliGenerator"
}

generators = {}

for model in models.keys():
    generators[model] = Generator(models[model], lite=True)
print(generators)

def load_models():
    for generator in generators.values():
        generator.load_model()

def load_model(model):
    generators[model].load_model()

def generate(image, style):
    return generators[style].generate(image)

def is_loaded(model):
    return generators[model].is_loaded()
    
#load_models()