import pytest
import os
import numpy as np
from PIL import Image

# get image data
@pytest.fixture
def get_images():
    image_dir = './tests/test_images'

    # it is assumed that the image name is the vegetable type
    images = os.listdir(image_dir)

    # open all of them and return a nested list, where the first list is Image objects and the second list is their respective labels
    return [[Image.open(f'{image_dir}/{img_path}') for img_path in images], [label.split('.')[0] for label in images]]

# lambda function to transform and scale so it can apply to two fixtures
# unfortunately, I can't call the same fixture twice with different indirect parameters
pixelScaler = np.vectorize(lambda x: x / 255)
transformer = lambda img, pixels: pixelScaler(np.array(img.convert('L').resize((pixels, pixels)))).reshape(1, pixels, pixels, 1)

# transform the images and scale to 128 pixels
@pytest.fixture
def transform_images_128(get_images):
    return [[transformer(img, 128) for img in get_images[0]], get_images[1]]

# same but with 31 pixels
@pytest.fixture
def transform_images_31(get_images):
    return [[transformer(img, 31) for img in get_images[0]], get_images[1]]