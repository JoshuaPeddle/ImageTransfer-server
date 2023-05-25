

def test_random_image(client):
    response = client.get("/random")
    assert response.status_code == 200
    assert b"https://raw.githubusercontent.com/JoshuaPeddle/ImageTransfer-server/master/random_images" in response.data
    assert b".jpg" in response.data

def test_styles(client):
    response = client.get("/styles")
    assert response.status_code == 200
    assert b"mini" in response.data
    assert b"full_name" in response.data
    assert b"num_variants" in response.data