import tensorflow as tf
from models.utils import tensor_to_image
import tensorflow_hub as hub
import numpy as np
import os
import functools
from models.utils import crop_center
from random import randint
from time import sleep
## Design a general class for generators so that we can easily add new models using a config file
class FastGenerator():

    def __init__(self, styles: dict):
        self.model = None
        self.styles = styles
        self.style_images = {}

    def is_loaded(self):
        return self.model is not None
    
    def load_style_images(self):
        for key, value in self.styles.items():
            urls = value['style_images']
            if type(urls) is not list:
                val = self.load_image(value)
                self.style_images[key] = val
                return 
            else:
                self.style_images[key] = []
                for url in urls:
                    val = self.load_image(url,_sleep=True)
                    if val is not None:
                        self.style_images[key].append(val)


        for key, value in self.style_images.items():
            self.style_images[key] = [tf.nn.avg_pool(item, ksize=[3,3], strides=[1,1], padding='SAME') for item in value]
            print('pooling')
             
    
    def load_model(self):
        hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
        hub_module = hub.load(hub_handle)
        self.model = hub_module
        self.load_style_images()
        return True
    
    @functools.lru_cache(maxsize=None)
    def load_image(self, image_url,image_size=(256, 256), preserve_aspect_ratio=True, _sleep=False ):
        """Loads and preprocesses images."""
        # Cache image file locally.
        try :
            image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
        except:
            print('failed to load image')
            return None
        # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
        img = tf.io.decode_image(
            tf.io.read_file(image_path),
            channels=3, dtype=tf.float32)[tf.newaxis, ...]
        img = crop_center(img)
        img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
        if _sleep:
            print('sleep')
            sleep(1)
        return img

    def generate(self, image, style):
        image = np.asarray(image)
        ## Normalize the image
        if image.shape[-1] == 4:
            image = image[...,:-1]
        image = (tf.cast(image, tf.float32) / 127.5) - 1

        image = tf.expand_dims(image, 0)
        style_image = self.style_images[style][randint(0, len(self.style_images[style])-1)]


        outputs = self.model(tf.constant(image), tf.constant(style_image))
        stylized_image = outputs[0]
        print(stylized_image.shape)
        stylized_image = tf.squeeze(stylized_image)
        stylized_image = stylized_image*255.0
        return tensor_to_image(stylized_image)
    