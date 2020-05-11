
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from pytorch_models.smaller_model import Model
#from pytorch_models.residual_model128 import Model
#from pytorch_models.residual_model256 import Model

import torch
import torch.utils.data
import torch.nn as nn
from torchvision import transforms


haarcascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")


def create_video(class_: str="0", video_file: str="", decoder: str="", face_dim=(256, 256), model_path: str=""):
    model = Model().cuda()
    model.load_state_dict(torch.load(model_path))

    count = 0
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        _, frame = cap.read()
        image = frame

        if class_ == "0":
            pass
        elif class_ == "1":
            image = cv2.resize(image, (874, 437))

        try:
            # finds face
            gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = haarcascade.detectMultiScale(gray_scale_image, 1.3, 5)[0]
            x, y, w, h = face[0], face[1], face[2], face[3]

            # corrections for target images:
            if class_ == "0":
                x += 50
                y += 80
                
                w -= 85
                h -= 85
            
            # corrections for source images
            if class_ == "1":
                x += 30
                y += 50

                w -= 45
                h -= 45

            # warmer image temperature of the source image because of skin tone differences
            if class_ == "0":
                image = change_temperature(image, temp=4500)
                image = change_brightness(image, c=0.65)

                image[:, :, 0] += 10
            
            # crop out face, resize it, normalize it to [0; 1], convert to tensor
            face = image[y:(y + h), x:(x + w)]
            face_ = cv2.resize(face, face_dim)

            # run face through encoder and given decoder
            pil_reproduced_ = model_reproduction(model, face_, decoder=decoder)
            pil_reproduced = pil_reproduced_.resize((w, h))

            # convert it to RGB for OpenCV
            pil_reproduced = pil_bgr2rgb(pil_reproduced)

            # pate the reproduced face on the video frame
            image = Image.fromarray(np.array(image))
            image.paste(pil_reproduced, (x, y))

            image.paste(Image.fromarray(face_), (10, 10))
            image.paste(pil_bgr2rgb(pil_reproduced_), (148, 10))
            image = np.array(image)

        except Exception as e:
            print(e)

        cv2.imshow("img", image)
        cv2.waitKey(1)
        
        if cv2.waitKey(25) == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# get model reproduction
def model_reproduction(model, face, decoder: str=""):
    face = np.array(face) / 255
    face = torch.Tensor(face).cuda().reshape(1, 3, 128, 128)

    # reproduce image with autoencoder
    reproduced = model.eval()(face, label=decoder, mode="test")[0]

    # detach from pytorch-graph, resize it to original shape
    reproduced = reproduced.cpu().detach().numpy().reshape(128, 128, 3) * 255
    pil_reproduced = Image.fromarray(reproduced.astype("uint8"))

    return pil_reproduced


# convert BGR to RGB
def pil_bgr2rgb(pil_image):
    b, g, r = pil_image.split()
    pil_image = Image.merge("RGB", (r, g, b))

    return pil_image


# change image brightness
def change_brightness(cv_image, c: float=1):
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)

    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced_im = enhancer.enhance(c)

    cv_image = np.array(enhanced_im)
    cv_image = cv_image[:, :, ::-1].copy()

    return cv_image


# change image temperature
def change_temperature(cv_image, temp: int=1000):
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)

    kelvin_table = {
        1000: (255,56,0),
        1500: (255,109,0),
        2000: (255,137,18),
        2500: (255,161,72),
        3000: (255,180,107),
        3500: (255,196,137),
        4000: (255,209,163),
        4500: (255,219,186),
        5000: (255,228,206),
        5500: (255,236,224),
        6000: (255,243,239),
        6500: (255,249,253),
        7000: (245,243,255),
        7500: (235,238,255),
        8000: (227,233,255),
        8500: (220,229,255),
        9000: (214,225,255),
        9500: (208,222,255),
        10000: (204,219,255)
    }

    r, g, b = kelvin_table[temp]
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    
    pil_image = pil_image.convert("RGB", matrix)
    cv_image = np.array(pil_image)
    cv_image = cv_image[:, :, ::-1].copy()

    return cv_image


if __name__ == "__main__":
    target_label = 0
    source_label = 1

    target_video = "datasets/videos/target_video1.mp4"
    source_video = "datasets/videos/micheal_scott/source_video1.mp4"

    model_path4 = "models/four/four_model.pt"
    model_path3 = "models/three/three_model.pt"
    model_path6 = "models/six/six_model.pt"
    model_path7 = "models/seven/seven_model.pt"

    create_video(class_="1", video_file=source_video, decoder="0", face_dim=(128, 128), model_path=model_path6)
    #create_video(class_="0", video_file=target_video, decoder="1", face_dim=(128, 128), model_path=model_path6)
