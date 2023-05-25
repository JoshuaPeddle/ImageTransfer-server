from pathlib import Path
from PIL import Image
from io import BytesIO

def test_random_image(client):
    response = client.get("/random")
    assert response.status_code == 200
    assert b"https://raw.githubusercontent.com/JoshuaPeddle/\
ImageTransfer-server/master/random_images" in response.data
    assert b".jpg" in response.data

def test_styles(client):
    response = client.get("/styles")
    assert response.status_code == 200
    assert b"mini" in response.data
    assert b"full_name" in response.data
    assert b"num_variants" in response.data

def test_generate_variant(client):
    test_image_src = Path(__file__).parent/"test_image.jpg"
    image =Image.open(test_image_src)
    response = client.post("/generate/monet/1", data={
        'image': (test_image_src).open('rb'),
    })
    assert response.status_code == 200
    assert b"JFIF" in response.data
    res_image = Image.open(BytesIO(response.data))
    assert res_image.size == image.size

