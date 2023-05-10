import os
import time
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
from numpy import asarray
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

def tensor_to_image(tensor):
    tensor = tf.cast(tf.clip_by_value(tensor, 0, 255), tf.uint8)
    tensor = Image.fromarray(tensor.numpy())
    tensor.save('upscaled_image.jpg')
    return 'upscaled_image.jpg'


def request_to_image(request):
    return asarray(Image.open(request.files['image']))

def _monet(image, upscale=False):
    IMAGE_SIZE = (256, 256)
    def decode_image(image):
        #image = tf.image.decode_jpeg(image, channels=3)
        image = (tf.cast(image, tf.float32) / 127.5) - 1
        image = tf.reshape(image, [*IMAGE_SIZE, 3])
        return image

    new_model = tf.keras.models.load_model('models/monet_generator', compile=False)
    #new_model.summary()
    image = tf.image.resize(image, IMAGE_SIZE)
    image = decode_image(image)
    
    image = tf.expand_dims(image, 0)
    start = time.time()
    prediction = new_model(image, training=False)
    prediction = tf.reshape(prediction, [256, 256, 3])
    prediction = (prediction + 1) / 2
    prediction = tf.image.convert_image_dtype(prediction, tf.uint8)
    print("Time Taken: %f" % (time.time() - start))

    if upscale:
        return _upscale(prediction)
    
    return tensor_to_image(prediction)




def _upscale(image):
    SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
    def preprocess_image(image_path):
        """ Loads image from path and preprocesses to make it model ready
            Args:
                image_path: Path to the image file
        """
        #hr_image = tf.image.decode_image(tf.io.read_file(image_path))
        hr_image = image
        # If PNG, remove the alpha channel. The model only supports
        # images with 3 color channels.
        if hr_image.shape[-1] == 4:
            hr_image = hr_image[...,:-1]
        hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
        hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
        hr_image = tf.cast(hr_image, tf.float32)
        return tf.expand_dims(hr_image, 0)


    hr_image = preprocess_image(image)
    model = hub.load(SAVED_MODEL_PATH)

    start = time.time()
    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)
    print("Time Taken: %f" % (time.time() - start))
    fake_image = tf.image.resize(fake_image, (512,512))
    return tensor_to_image(fake_image)