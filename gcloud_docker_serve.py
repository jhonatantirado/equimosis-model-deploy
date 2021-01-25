import base64
import io
import json

import requests

# Tutorial at: https://cloud.google.com/vision/automl/docs/containers-gcs-tutorial?hl=en_US#top_of_page
# Export the trained model, download it from Google Cloud, and copy it to a local path: /Users/av_jtirado/Documents/DatacionEquimosis/TesisDatacionEquimosis/best-model
# Install Docker Desktop

# Set environment variables
# export bruisedating_container=equimosisv3_20191108024659
# export bruisedating_port=5408 -- This is the localhost port

# export bruisedating_model_path=/Users/av_jtirado/Documents/DatacionEquimosis/TesisDatacionEquimosis/best-model
# export CPU_DOCKER_GCR_PATH=gcr.io/cloud-devrel-public-resources/gcloud-container-1.14.0:latest -- This is the Docker image

# Pull the Docker image
# sudo docker pull ${CPU_DOCKER_GCR_PATH}

# Run container in Docker
# sudo docker run --rm --name ${bruisedating_container} -p ${bruisedating_port}:8501 -v ${bruisedating_model_path}:/tmp/mounted_model/0001 -t ${CPU_DOCKER_GCR_PATH}
# A saved_model.pb file should exist in the bruisedating_model_path directory

# Execute this script
# python gcloud_docker_serve.py

# Expected result:
# {'predictions': [{'scores': [0.00662456034, 0.960488379, 0.00662456034, 0.00662456034, 0.0047885091, 0.00822485238, 0.00662456034], 
# 'labels': ['ThreeDays', 'TwoDays', 'MoreThanSeventeenDays', 'TwelveDays', 'SixDays', 'SeventeenDays', 'NoBruise'], 
# 'key': 'editada_WhatsAppImage2019-09-15at9.33.47PM'}]}

def container_predict(image_file_path, image_key, port_number=5408):
    """Sends a prediction request to TFServing docker container REST API.

    Args:
        image_file_path: Path to a local image for the prediction request.
        image_key: Your chosen string key to identify the given image.
        port_number: The port number on your device to accept REST API calls.
    Returns:
        The response of the prediction request.
    """

    with io.open(image_file_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    # The example here only shows prediction with one image. You can extend it
    # to predict with a batch of images indicated by different keys, which can
    # make sure that the responses corresponding to the given image.
    instances = {
            'instances': [
                    {'image_bytes': {'b64': str(encoded_image)},
                     'key': image_key}
            ]
    }

    # This example shows sending requests in the same server that you start
    # docker containers. If you would like to send requests to other servers,
    # please change localhost to IP of other servers.
    url = 'http://localhost:{}/v1/models/default:predict'.format(port_number)
    print (url)
    print (instances)

    response = requests.post(url, data=json.dumps(instances))
    print(response.json())

if __name__ == '__main__':
    #image_file_path='/Users/av_jtirado/Documents/DatacionEquimosis/TesisDatacionEquimosis/equimosisv3/Training/TwoDays/editada_WhatsAppImage2019-09-15at9.33.47PM.jpeg'
    image_file_path='/Users/av_jtirado/Desktop/editada_IMG-20191018-WA0031.jpg'
    image_key='Screen Shot 2020-12-21 at 18.03.14.PNG'
    container_predict(image_file_path, image_key, port_number=5408)