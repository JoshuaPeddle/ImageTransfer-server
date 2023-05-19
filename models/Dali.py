
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
import time 
from models.utils import tensor_to_image

from PIL.ImageOps import fit
from numpy import asarray
class DaliGenerator():

    def __init__(self):
        self.model = None

    def is_loaded(self):
        return self.model is not None

    def load_model(self):
        self.model =  from_pretrained_keras("JoshuaPeddle/DaliGenerator", compile=False)

    def generate(self, image):

        IMAGE_SIZE = (256, 256)
        def decode_image(image):
 
            image = asarray(fit(image, IMAGE_SIZE))
            if image.shape[-1] == 4:
                image = image[...,:-1]
            image = (tf.cast(image, tf.float32) / 127.5) - 1
            return image
        
        image = decode_image(image)
        
        image = tf.expand_dims(image, 0)
        start = time.time()
        prediction = self.model(image, training=False)
        prediction = tf.reshape(prediction, [256, 256, 3])
        prediction = (prediction + 1) / 2
        prediction = tf.image.convert_image_dtype(prediction, tf.uint8)
        print("Time Taken: %f" % (time.time() - start))

        
        return tensor_to_image(prediction)
