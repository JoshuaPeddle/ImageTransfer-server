import os
from random import randint
URL_ROOT = "https://raw.githubusercontent.com/JoshuaPeddle/ImageTransfer-server/master/random_images/"


# Get a list of filenames from random_images folder, these corrospond to the
#  images that will be used for the random route
def load_image_names():
    return os.listdir('./random_images')


random_images = load_image_names()

def generate_random_image_url():    
    return URL_ROOT + random_images[randint(0, len(random_images) - 1)]
    

# Path: models/models.py
