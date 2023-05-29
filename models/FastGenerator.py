import tensorflow as tf
from .utils import tensor_to_image
from .utils import crop_center
import tensorflow_hub as hub
import numpy as np
import os
import functools
from random import randint
from time import sleep


## Design a general class for generators so that we can
# easily add new models using a config file
class FastGenerator:
    def __init__(self, styles: dict):
        self.model = None
        self.styles = styles
        self.style_images = {}
        self.image = None

    def is_loaded(self):
        return self.model is not None

    def load_style_images(self):
        for key, value in self.styles.items():
            urls = value["style_images"]
            if type(urls) is not list:
                val = self.load_image(value)
                self.style_images[key] = val
                return
            else:
                self.style_images[key] = []
                for url in urls:
                    val = self.load_image(url, _sleep=True)
                    if val is not None:
                        self.style_images[key].append(val)
        for key, value in self.style_images.items():
            self.style_images[key] = [
                tf.nn.avg_pool(item, ksize=[3, 3], strides=[1, 1], padding="SAME")
                for item in value
            ]
            print("pooling")

    def load_model(self):
        self.load_style_images()
        hub_handle = "https://tfhub.dev/google/magenta/\
arbitrary-image-stylization-v1-256/2"
        hub_module = hub.load(hub_handle)
        self.model = hub_module
        return True

    def load_image(
        self,
        image_url,
        image_size=(256, 256),
        preserve_aspect_ratio=True,
        _sleep=False,
        retries=5,
    ):
        """Loads and preprocesses images."""
        try:
            image_path = tf.keras.utils.get_file(
                os.path.basename(image_url)[-128:], image_url
            )
            img = tf.io.decode_image(
                tf.io.read_file(image_path), channels=3, dtype=tf.float32
            )[tf.newaxis, ...]
            img = tf.image.resize(
                img,
                image_size,
                preserve_aspect_ratio=True,
                antialias=False,
                method="lanczos3",
            )
            img = crop_center(img)
            if _sleep:
                # print('sleep')
                sleep(0.1)
            return img
        except (ValueError, Exception):
            if retries > 0:
                sleep(2)
                return self.load_image(
                    image_url, image_size, preserve_aspect_ratio, _sleep, retries - 1
                )
            print("failed to load image")
            return None

    @functools.lru_cache(maxsize=100)
    def preprocess_image(self, uuid, premultiply):
        image = self.image
        original_shape = image.size
        original_shape = (original_shape[1], original_shape[0])
        ## If the image is larger than 384 in width or height, resize it down to 384.
        if max(original_shape) > 384:
            image = tf.image.resize(
                image,
                (384, 384),
                preserve_aspect_ratio=True,
                antialias=False,
                method="gaussian",
            )
        image = np.asarray(image)
        ## Drop the alpha channel if it exists
        if image.shape[-1] == 4:
            if premultiply:
                if image.shape[-1] == 4:
                    # Copy the image to avoid modifying it in place
                    image = image.copy()
                    # Premultiply the RGB channels by the alpha channel
                    image[..., :3] *= image[..., 3:4]
                    image = image[..., :3]  # Remove the alpha channel
            else:
                image = image[..., :-1]
        ## Normalize the image
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        image = tf.expand_dims(image, 0)
        return image, uuid, original_shape

    def generate(self, image, style, variant=None, premultiply=True, uuid=None):
        self.image = image
        image, uuid, original_shape = self.preprocess_image(uuid, premultiply)
        if variant is not None:
            if variant >= len(
                self.style_images[style]
            ):  # Handle case where server has less than reported number of styles
                variant = randint(0, len(self.style_images[style]) - 1)
            style_image = self.style_images[style][variant]
        else:
            style_image = self.style_images[style][
                randint(0, len(self.style_images[style]) - 1)
            ]
        outputs = self.model(tf.constant(image), tf.constant(style_image))
        stylized_image = outputs[0]
        stylized_image = tf.squeeze(stylized_image)
        stylized_image = stylized_image * 255.0
        if image.shape[1] != original_shape[0] or image.shape[2] != original_shape[1]:
            stylized_image = tf.image.resize(
                stylized_image,
                original_shape,
                preserve_aspect_ratio=False,
                method="lanczos3",
                antialias=True,
            )
        return tensor_to_image(stylized_image)
