import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import time 

SOURCE_IMAGE_PATH = 'test_image2.jpg'
API_TARGET = 'http://localhost:5001/generate/monet/1'

num_workers = 2
def load_source_image():
    return Image.open(SOURCE_IMAGE_PATH)

src_img = load_source_image()

def generate_image():
    response = requests.post(API_TARGET, files={'image': open(SOURCE_IMAGE_PATH, 'rb')})
    return response.content


start = time.time()
with ThreadPoolExecutor(max_workers=num_workers) as pool:
    futures = [pool.submit(generate_image) for _ in range(100)]
    for future in futures:
        print('.', end='')

end = time.time()
print(f"Time taken: {end - start}")