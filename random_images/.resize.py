# This script should resize images in new_random_images/ to 384 x 384
# Since it is ran before build and not at runtime quality can be emphasized over speed

import os
from PIL import Image

for filename in os.listdir('new_random_images'):
    if filename.endswith('.jpg'):
        im = Image.open('new_random_images/' + filename)
        im = im.resize((384, 256), Image.Resampling.LANCZOS )
        im.save('new_random_images/' + "new."+filename, quality=92)