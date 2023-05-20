import tensorflow as tf
from huggingface_hub import from_pretrained_keras, hf_hub_download
import time 
from models.utils import tensor_to_image
from PIL.ImageOps import fit
from numpy import asarray

class PicassoGenerator():


    def __init__(self):
        self.model = None

    def is_loaded(self):
        return self.model is not None
    def from_pretrained_tflite(self):
        model = hf_hub_download(repo_id="JoshuaPeddle/PicassoGeneratorLite", filename="model.tflite")
        self.model = tf.lite.Interpreter(model_path=model)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        print(self.input_details)
        print(self.output_details)
        print(self.model)
        print("Loaded Model")

    def load_model(self, lite=True):
        if lite:
            self.from_pretrained_tflite()
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

