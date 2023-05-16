from PIL import Image
import tensorflow as tf

def tensor_to_image(tensor):
    tensor = tf.cast(tf.clip_by_value(tensor, 0, 255), tf.uint8)
    tensor = Image.fromarray(tensor.numpy())
    tensor.save('upscaled_image.jpg')
    return 'upscaled_image.jpg'

