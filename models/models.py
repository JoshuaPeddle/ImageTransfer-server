from models.FastGenerator import FastGenerator
import json
from random import randint
from functools import cache

# Load style from ./styles.json
with open('models/styles.json') as f:
    styles = json.load(f)

generator = FastGenerator(styles)

def load_model():
    loaded = generator.load_model()
    return loaded
    
def generate(image, style, variant=None, uuid=None):
    return generator.generate(image, style=style, variant=variant, uuid=uuid)

# This return JSON. The key shoul dbe each style available, and the value should be a single URL corresponding to the first
# entry of styles.json for that style

def get_styles():
    styles = generator.styles
    to_return = {}
    # Currently the values of styles are a list of urls.
    # Only want the first value not in a list
    for style in styles:
        print(style)
        bg = styles[style]['mini']
        full_name = styles[style]['style_name']
        print(style, bg, full_name)
        to_return[style] = {}
        to_return[style]['mini'] = bg
        to_return[style]['full_name'] = full_name
        to_return[style]['num_variants'] = len(styles[style]['style_images'])
    return to_return

    
    