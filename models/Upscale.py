
import tensorflow as tf
import os
import time 
from models.utils import tensor_to_image
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
import tensorflow_hub as hub


class UpscaleGenerator():

    def load_model(self):
        self.model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

    def generate(self, image):

        def preprocess_image(_image):
            """ Loads image from path and preprocesses to make it model ready
                Args:
                    image_path: Path to the image file
            """
            hr_image = _image
            # If PNG, remove the alpha channel. The model only supports
            # images with 3 color channels.
            if hr_image.shape[-1] == 4:
                hr_image = hr_image[...,:-1]
            hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
            hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
            hr_image = tf.cast(hr_image, tf.float32)
            return tf.expand_dims(hr_image, 0)

        hr_image = preprocess_image(image)
        start = time.time()
        fake_image = self.model(hr_image)
        fake_image = tf.squeeze(fake_image)
        print("Time Taken: %f" % (time.time() - start))
        fake_image = tf.image.resize(fake_image, (512,512))
        return tensor_to_image(fake_image)

        

