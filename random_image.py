import os
from random import randint

URL_ROOT = "https://raw.githubusercontent.com/JoshuaPeddle/ImageTransfer-server/master/random_images/"


# Get a list of filenames from random_images folder, these corrospond to the
#  images that will be used for the random route
def load_image_names():
    return  [f for f in os.listdir("./random_images") if not f.startswith('.')]


random_images = load_image_names()


def generate_random_image_urls(n):
    n = int(n)
    if n > len(random_images):
        n = len(random_images)

    if n == 1:
        return URL_ROOT + random_images[randint(0, len(random_images) - 1)]

    urls = []
    for i in range(n):
        urls.append(URL_ROOT + random_images[randint(0, len(random_images) - 1)])
    return urls


# Path: models/models.py
