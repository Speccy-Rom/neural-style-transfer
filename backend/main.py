import uuid

import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image

import config
import inference

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome from the API"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get('/{style}')
def get_image(style: str, file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    model = config.STYLES[style]
    output, resized = inference.inference(model, image)
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    cv2.imwrite(name, output)
    return {"name": name}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
