import os
import requests
import base64
import json
from PIL import Image
from io import BytesIO
from imagekitio import ImageKit
from dotenv import load_dotenv

load_dotenv()

API_ENDPOINT = "https://upload.imagekit.io/api/v1/files/upload"
PRIVATE_KEY = os.getenv("IMAGEKIT_PRIVATE_KEY")  # replace with your actual private key
PUBLIC_KEY = os.getenv("IMAGEKIT_PUBLIC_KEY")  # replace with your actual public key
STYLE_FILE = "./models/styles.json"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
}

imagekit = ImageKit(
    private_key=PRIVATE_KEY,
    public_key=PUBLIC_KEY,
    url_endpoint="https://ik.imagekit.io/4adj1pc55",
)


def upload_file(file, file_name):
    with open(file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        result = imagekit.upload_file(
            file=encoded_string,  # required
            file_name=file_name,  # required
        )
        print(result.url)
        return result.url


def validate_url(
    url,
    artist,
):
    # Url should contain imagekit.io, artist name
    if "imagekit.io" not in url:
        return False
    if artist not in url:
        return False
    return True


# Load your style JSON
with open(STYLE_FILE, "r") as f:
    styles = json.load(f)

    # Save the updated JSON
with open(STYLE_FILE + ".old", "w") as f:
    json.dump(styles, f, indent=4)

for artist, data in styles.items():
    for i, url in enumerate(data["style_images"]):
        # Could already be an imagekit URL
        if validate_url(url, artist):
            continue
        try:
            # Download the image
            response = requests.get(url, headers=headers)
            img = Image.open(BytesIO(response.content))

            # Save the image to a temporary file
            img.save("temp.jpg")

            # Upload the image to S3
            new_key = f"{artist}/{i}.jpg"
            new_url = upload_file("temp.jpg", new_key)
            is_valid = validate_url(new_url, artist)
            if not is_valid:
                raise Exception(f"Invalid URL: {new_url}")

            # Update the URL in the styles JSON
            styles[artist]["style_images"][i] = new_url

            # Update the 'mini' URL if it matches the current URL
            if data["mini"] == url:
                styles[artist]["mini"] = new_url
        except Exception as e:
            print(f"Failed to process URL {url}: {e}")

# Save the updated JSON
with open(STYLE_FILE, "w") as f:
    json.dump(styles, f, indent=4)
