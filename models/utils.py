from PIL import Image
import tensorflow as tf

def tensor_to_image(tensor):
    tensor = tf.cast(tf.clip_by_value(tensor, 0, 255), tf.uint8)
    tensor = Image.fromarray(tensor.numpy())
    tensor.save('upscaled_image.jpg')
    return 'upscaled_image.jpg'

def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image