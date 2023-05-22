from models.FastGenerator import FastGenerator


IMAGE_SIZE = (512, 512)

model = None

styles = {
    'monet': 'https://uploads8.wikiart.org/images/claude-monet/haystack-at-giverny-1886.jpg!Large.jpg',
    'gogh': 'https://uploads4.wikiart.org/00142/images/vincent-van-gogh/the-starry-night.jpg!Large.jpg',
    'dali': 'https://uploads6.wikiart.org/images/salvador-dali/the-persistence-of-memory-1931.jpg!Large.jpg',
    'picasso': 'https://d3d00swyhr67nd.cloudfront.net/w800h800/collection/TATE/TATE/TATE_TATE_T05010_10-001.jpg',
    'kandinsky': 'https://uploads2.wikiart.org/images/wassily-kandinsky/moscow-i-1916.jpg!Large.jpg',
}

generator = FastGenerator(styles)

def load_model():
    loaded = generator.load_model()
    return loaded
    
def generate(image, style):
    return generator.generate(image, style)

def is_loaded(model):
    return generator.is_loaded()
