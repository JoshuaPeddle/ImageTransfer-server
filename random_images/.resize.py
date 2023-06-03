# This script should resize images in new_random_images/ to 384 x 384
# Since it is ran before build and not at runtime quality can be emphasized over speed

import os
from PIL import Image
from pathlib import Path

Path('/root/dir/sub/file.ext').stem
for filename in os.listdir('.'):
    if filename.endswith('.jpg'):
        im = Image.open('./' + filename)
        im = im.resize((384, 256), Image.Resampling.LANCZOS )
        new_filename = Path('./' + filename).stem
        print(new_filename)
        im.save('./' + "new."+new_filename+'.webp', quality=85, method=6)