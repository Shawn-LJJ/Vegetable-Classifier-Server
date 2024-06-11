# this test is to test connection to the models in the docker
import pytest 
from models import get_labels
from numpy import argmax
import requests
import json

# get the url that can input the pixel size model
# get_url_test = lambda pixel: f'http://localhost:8501/v1/models/model_{pixel}'  # url for locally deployed to test connectivity
# get_url_predict = lambda pixel: f'http://localhost:8501/v1/models/model_{pixel}:predict'  # url for locally deployed to predict
get_url_test = lambda pixel: f'https://twob01-2239745-shawnlim-ca2-models.onrender.com/v1/models/model_{pixel}'    # url for render deployed test
get_url_predict = lambda pixel: f'https://twob01-2239745-shawnlim-ca2-models.onrender.com/v1/models/model_{pixel}:predict'    # url for render deployed predict

LABELS = get_labels()

# function to post the data and get the prediction and return the predicted label
def post_prediction(img, pixel):
    data = json.dumps({'signature_name' : 'serving_default', 'instances' : img.tolist()})
    headers = {"content-type": "application/json"}
    url = get_url_predict(pixel)

    res = requests.post(url, data, headers=headers)
    pred = json.loads(res.text)['predictions']
    return LABELS[argmax(pred)].lower()

# test the server by sending a get request and extract the error code message from the response
def test_connection():
    pixels = [128, 31]

    for pixel in pixels:
        header = {"content-type": "application/json"}

        url = get_url_test(pixel)
        res = requests.get(url=url, headers=header)
        error_code = json.loads(res.text)['model_version_status'][0]['status']['error_code']
        assert error_code == 'OK'

# test both the model 128 and model 31 
def test_url_prediction(transform_images_128, transform_images_31):

    for i, img in enumerate(transform_images_128[0]):
        pred = post_prediction(img, 128)
        assert pred == transform_images_128[1][i]
    
    for i, img in enumerate(transform_images_31[0]):
        pred = post_prediction(img, 31)
        assert pred == transform_images_31[1][i]