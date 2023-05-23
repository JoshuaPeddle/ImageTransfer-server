# Image Transfer and Upscaling API
This service provides APIs to perform image upscaling and style transfer (Monet style) using Machine Learning models. The project is built using Flask, a lightweight WSGI web application framework.


```bash
git clone <repository-url>
cd <project-directory>
Install the dependencies using pip.
```
```bash
pip install -r requirements.txt
```

## Running the Application
Run the application with the following command:

```bash
 python3 -m flask run --host=0.0.0.0
```
The application should start and be accessible at localhost:5000.

## API Endpoints
The application exposes two API endpoints:

```/monet```: POST endpoint which accepts an image file and returns the image stylized in the style of Monet paintings.

```/upscale```: POST endpoint which accepts an image file and returns the upscaled version of the image.
#File Structure
```app.py```: This is the main application file which runs the Flask application and defines the API endpoints.
```models/models.py```: This file contains the model loading and prediction functions.
## models/models.py
```tensor_to_image(tensor)```: Converts a TensorFlow tensor to an image and saves it to 'upscaled_image.jpg'.

```request_to_image(request)```: Converts a Flask request object containing an image to a numpy array.

```_monet(image, upscale=False)```: Accepts an image as input and returns the image stylized in the style of Monet paintings.

```_upscale(image)```: Accepts an image as input and returns the upscaled version of the image.
## Sending Requests to the API
For both endpoints, the image file should be included in the request as form data with the key 'image'. Here is an example using curl:

```bash
Copy code
curl -X POST -F "image=@<image-file-path>" http://localhost:5000/monet
```
```bash
Copy code
curl -X POST -F "image=@<image-file-path>" http://localhost:5000/upscale
Replace <image-file-path> with the path to your image file.
```
## Notes
The model files for the Monet style transfer (models/monet_generator) are not included in this repository. You need to download them separately and place them in the correct location.
Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT

docker buildx build --platform linux/amd64,linux/arm64 -t joshuapeddle/imagetransfer-server:0.0.2 --push .

docker run -p 5000:5000 joshuapeddle/imagetransfer-server:0.0.2



## Imagekit upload styles
```bash
pip install Imagekit
```
```bash
python imagekit_upload.py
```
