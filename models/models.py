from models.FastGenerator import FastGenerator
import json


# Load style from ./styles.json
with open('models/styles.json') as f:
    styles = json.load(f)

generator = FastGenerator(styles)

def load_model():
    loaded = generator.load_model()
    return loaded
    
def generate(image, style):
    return generator.generate(image, style)

