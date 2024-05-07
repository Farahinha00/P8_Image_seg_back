from fastapi import FastAPI
from urllib.request import urlretrieve
from time import sleep
import os
import tensorflow as tf

app = FastAPI()

model = None
# Define the URL and the filename
url1 = "https://p8och5.blob.core.windows.net/modelevgg/model_vgg_unet.h5"
filename1 = "modelvggunet.h5"
# Download the file
urlretrieve(url1, filename1)


def download_file(url, filename):
    if not os.path.exists(filename):
        urlretrieve(url, filename)
        while not os.path.exists(filename) or os.path.getsize(filename) == 0:
            sleep(1)  # Attendre 1 seconde avant de vérifier à nouveau

def init():
    global model

    download_file("https://p8och5.blob.core.windows.net/modelevgg/model_vgg_unet.h5", filename1)

    model = tf.keras.models.load_model(filename1)

# Événement de démarrage pour exécuter la fonction init
@app.on_event("startup")
async def startup_event():
    init()

    

@app.get("/")
async def root():
    return {"message": "Hello World"}
