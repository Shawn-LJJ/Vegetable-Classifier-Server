# this test is to simply only test whether the prediction works
import pytest 
import models
from numpy import argmax

# @pytest.mark.parametrize("transform_images", ["a", "b"], indirect=True)
def test_prediction(transform_images_128, transform_images_31):

    # instantiate the models
    model_128 = models.create_model_128()
    model_31 = models.create_model_31()
    labels = models.get_labels()

    # for 128 pixels, make prediction for each image
    for i, img in enumerate(transform_images_128[0]):
        pred = labels[argmax(model_128.predict(img, verbose=0))]
        assert pred.lower() == transform_images_128[1][i].lower()
    
    # same but for 31 pixels
    for i, img in enumerate(transform_images_31[0]):
        pred = labels[argmax(model_31.predict(img, verbose=0))]
        assert pred.lower() == transform_images_31[1][i].lower()