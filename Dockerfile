FROM tensorflow/serving

COPY / /setup

ENV CONFIG_FILE=/setup/config/model.config MODELS_BASE_PATH=/models PORT=8501

EXPOSE 8501

# update APT and install python packages
RUN apt-get update && apt-get -y upgrade && apt-get install -y python3 python3-pip
# upgrade pip and install the necessary packages
RUN pip install --upgrade pip && pip install -r /setup/requirements.txt
# run the python file to build the models
RUN python3 /setup/model.py
# make the script for the tensorflow model server and make it executable
RUN echo '#!/bin/bash \n\n\
tensorflow_model_server \
--rest_api_port=$PORT \
--model_config_file=${CONFIG_FILE} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh && chmod +x /usr/bin/tf_serving_entrypoint.sh