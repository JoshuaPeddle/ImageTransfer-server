
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
import time 
from models.utils import tensor_to_image

from PIL.ImageOps import fit
from numpy import asarray
class PicassoGenerator():

    def load_model(self):
        self.model =  from_pretrained_keras("JoshuaPeddle/PicassoGenerator", compile=False)

    def generate(self, image):

        IMAGE_SIZE = (256, 256)
        def decode_image(image):
 
            image = asarray(fit(image, IMAGE_SIZE))
            if image.shape[-1] == 4:
                image = image[...,:-1]
            image = (tf.cast(image, tf.float32) / 127.5) - 1
            
            #image = tf.reshape(image, [*IMAGE_SIZE, 3])
            return image
        #image = image.filter(ImageFilter.GaussianBlur(radius=2))
        #image = tf.image.resize(image, IMAGE_SIZE)
        
        image = decode_image(image)
        
        image = tf.expand_dims(image, 0)
        start = time.time()
        prediction = self.model(image, training=False)
        prediction = tf.reshape(prediction, [256, 256, 3])
        prediction = (prediction + 1) / 2
        prediction = tf.image.convert_image_dtype(prediction, tf.uint8)
        print("Time Taken: %f" % (time.time() - start))

        
        return tensor_to_image(prediction)

