import models
import os
from time import time

# check it the environment for base model path exists, or else it will be created in this directory
BASE_PATH = os.getenv('MODELS_BASE_PATH')
if not BASE_PATH: BASE_PATH = '.'
print(f'Building models at {BASE_PATH}')

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# create the models 
model_128 = models.create_model_128()
model_31 = models.create_model_31()

# get the current time as the folder name and save the models
t = int(time())
model_128.save(f'{BASE_PATH}/model_128/{t}/')
model_31.save(f'{BASE_PATH}/model_31/{t}')