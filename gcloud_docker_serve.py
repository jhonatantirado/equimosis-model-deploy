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

# export bruisedating_model_path=/Users/jhonatantirado/code/equimosis-model-deploy/best-model
# export CPU_DOCKER_GCR_PATH=gcr.io/cloud-devrel-public-resources/gcloud-container-1.14.0:latest -- This is the Docker image

# Pull the Docker image
# sudo docker pull ${CPU_DOCKER_GCR_PATH}

# Run container in Docker
# sudo docker run --rm --name ${bruisedating_container} -p ${bruisedating_port}:8501 -v ${bruisedating_model_path}:/tmp/mounted_model/0001 -t ${CPU_DOCKER_GCR_PATH}


# sudo docker run --rm --name equimosisv3_20191108024659 -p 5408:8501 -v /Users/jhonatantirado/code/equimosis-model-deploy/best-model:/tmp/mounted_model/0001 -t gcr.io/cloud-devrel-public-resources/gcloud-container-1.14.0:latest

#Create docker image with base image plus prediction model, after running base image plus model path with docker run
#docker commit 35027f92f2f8(container ID) bruise-dating:v1.0
#docker commit 1aac7b1cd4014f4abac7f4ec447d4c70190d8e8f5f19126a3e980a6865cdd29f bruise-dating:v1.0

#Push image to Google Cloud Registry
#docker tag bruise-dating:v1.0 gcr.io/equimosis/bruise-dating-api-tf
#docker push gcr.io/equimosis/bruise-dating-api-tf

#Push image to new GOOGLE ARTIFACT Registry
#docker tag bruise-dating:v1.0 us-central1-docker.pkg.dev/inner-orb-400721/quickstart-docker-repo/bruise-dating:v1.0
#docker push us-central1-docker.pkg.dev/inner-orb-400721/quickstart-docker-repo/bruise-dating:v1.0


#Google Cloud SDK - deployment using app.yaml with custom runtime when using containers
#gcloud app deploy --image-url gcr.io/equimosis/bruise-dating-api-tf


#ARTIFACT REGISTRY - NEW WAY
#gcloud app deploy --image-url us-central1-docker.pkg.dev/inner-orb-400721/quickstart-docker-repo/bruise-dating:v1.0


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
    image_file_path='/Users/jhonatantirado/code/bruise-dating-paper-dataset/test/SixDays/e279b98b-10cb-4891-9f17-676b684ea845.jpg'
    image_key='e279b98b-10cb-4891-9f17-676b684ea845.jpg'
    container_predict(image_file_path, image_key, port_number=5408)