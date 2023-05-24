import tensorflow as tf
from models.utils import tensor_to_image
import tensorflow_hub as hub
import numpy as np
import os
import functools
from models.utils import crop_center
from random import randint
from time import sleep
import copy

## Design a general class for generators so that we can easily add new models using a config file
class FastGenerator():

    def __init__(self, styles: dict):
        self.model = None
        self.styles = styles
        self.style_images = {}
        self.image = None
        self.lite = True
        self.lite_model = None

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
            #self.style_images[key] = [tf.nn.avg_pool(item, ksize=[3,3], strides=[1,1], padding='SAME') for item in value]
            print('pooling')
             
    def load_model(self):
        self.load_style_images()
        if self.lite:
            lite_model =  tf.keras.utils.get_file('style_transform2.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite')
            self.lite_model = tf.lite.Interpreter(model_path=lite_model, num_threads=min(os.cpu_count(), 4))
            self.lite_model.allocate_tensors()
            self.style_predict_lite_setup()
            return True
        hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
        hub_module = hub.load(hub_handle)
        self.model = hub_module
        return True
    
    def style_predict_lite_setup(self):
        style_predict_path = tf.keras.utils.get_file('style_predict2.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite')
         # Load the model.
        interpreter = tf.lite.Interpreter(model_path=style_predict_path)
        # Set model input.
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        # For ever style image process it using the below code and save the bottleneck in place of the image
        for key, value in self.style_images.items():
            for i, style_image in enumerate(value):
                print(style_image)
                interpreter.set_tensor(input_details[0]["index"], style_image)
                # Calculate style bottleneck.
                interpreter.invoke()
                output_details = interpreter.get_output_details()[0]
                style_bottleneck = interpreter.get_tensor(output_details["index"])
                self.style_images[key][i] = copy.copy(style_bottleneck)
    

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
            #print('sleep')
            sleep(0.05)
        return img
    
    @functools.lru_cache(maxsize=20)
    def preprosess_image(self, uuid, premultiply):
        image = self.image
        image = np.asarray(image)
        ## Drop the alpha channel if it exists
        if image.shape[-1] == 4:
            if premultiply:
                if image.shape[-1] == 4:
                    image = image.copy() # Copy the image to avoid modifying it in place
                    image[..., :3] *= image[..., 3:4] # Premultiply the RGB channels by the alpha channel
                    image = image[..., :3] # Remove the alpha channel
            else:
                image = image[...,:-1]
        ## Normalize the image
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        image = tf.expand_dims(image, 0)
        return image, uuid
    

    def generate(self, image, style, variant=None, premultiply=True, uuid=None):
        if (self.lite):
            return self.generate_lite(image, style, variant, premultiply, uuid)
        self.image = image
        image, uuid = self.preprosess_image(uuid, premultiply)
        if variant is not None:
            style_image = self.style_images[style][variant]
        else:
            style_image = self.style_images[style][randint(0, len(self.style_images[style])-1)]
        outputs = self.model(tf.constant(image), tf.constant(style_image))
        stylized_image = outputs[0]
        stylized_image = tf.squeeze(stylized_image)
        stylized_image = stylized_image * 255.0
        print(self.preprosess_image.cache_info())
        return tensor_to_image(stylized_image)
    
    @functools.lru_cache(maxsize=20)
    def preprosess_lite(self, uuid, premultiply):
        
        image = self.image
        original_shape = image.size
        original_shape = (original_shape[1], original_shape[0])
        image = tf.image.resize(image, (384, 384), method='lanczos5', antialias=True)
        image = np.asarray(image)

        ## Drop the alpha channel if it exists
        if image.shape[-1] == 4:
            if premultiply:
                if image.shape[-1] == 4:
                    image = image.copy() # Copy the image to avoid modifying it in place
                    image[..., :3] *= image[..., 3:4] # Premultiply the RGB channels by the alpha channel
                    image = image[..., :3] # Remove the alpha channel
            else:
                image = image[...,:-1]
        ## Normalize the image to 0,1
        image = (tf.cast(image, tf.float32) / 255.0)
        image = tf.expand_dims(image, 0)
        return image, uuid, original_shape
    

    def generate_lite(self, image, style, variant=None, premultiply=True, uuid=None):
        print('lite')
        self.image = image
        # Resize to 384 x 384
        image, uuid, original_shape = self.preprosess_lite(uuid, premultiply)
        print(image.shape)
        if variant is not None:
            style_image = self.style_images[style][variant]
        else:
            style_image = self.style_images[style][randint(0, len(self.style_images[style])-1)]
        self.lite_model.set_tensor(self.lite_model.get_input_details()[0]['index'], image)
        self.lite_model.set_tensor(self.lite_model.get_input_details()[1]['index'], style_image)
        self.lite_model.invoke()
        stylized_image = self.lite_model.get_tensor(self.lite_model.get_output_details()[0]['index'])
        stylized_image = tf.squeeze(stylized_image)
        stylized_image = stylized_image * 255.0
        stylized_image = tf.image.resize(stylized_image, original_shape[0:2], method='mitchellcubic', antialias=True)
        print(original_shape)
        return tensor_to_image(stylized_image)