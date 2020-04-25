
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import time
import os
import json


haarcascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
target_video = "datasets/videos/target_video.mp4"
source_video = "datasets/videos/micheal_scott/source_video.mp4"
frame_information_file = "datasets/frame_boxes.json"


def save_coordinates(class_, frame: int=0, bbox: list=[]):
    with open(frame_information_file, "r") as f:
        content = json.load(f)

    new_frame = [class_, frame, bbox]
    content.append(new_frame)

    with open(frame_information_file, "w") as f:
        json.dump(content, f, indent=4)

def face_extraction(class_: int=0, video_file: str="", face_dim=(256, 256), save_to: str="", show: bool=False):
    count = 0
    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        _, frame = cap.read()
        image = frame

        try:
            gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # finds face
            face = haarcascade.detectMultiScale(gray_scale_image, 1.3, 5)[0]
            x, y, w, h = face[0], face[1], face[2], face[3]

            # corrections for target images
            if save_to.split("/")[-1] == "0":
                x += 50
                y += 80
                
                w -= 85
                h -= 85
            
            # corrections for source images
            if save_to.split("/")[-1] == "1":
                x += 45
                y += 80

                w -= 65
                h -= 65

            # warmer image temperature of the source image because of skin tone differences
            if save_to.split("/")[-1] == "0":
                image = change_temperature(image, temp=4500)
                image = change_brightness(image, c=0.65)

                image[:, :, 0] += 10

            # if the face is smaller than ~100 pixel, something must have went wrong
            if h > 100:                
                if show:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
                    cv2.imshow("img", image)
                    cv2.waitKey(2)
                # dont save faces if show-mode is on
                else:
                    save_coordinates(class_=class_, frame=cap.get(cv2.CAP_PROP_POS_FRAMES), bbox=[int(x), int(y), int(w), int(h)])

                    roi = image[y:(y + h), x:(x + w)]
                    roi = cv2.resize(roi, face_dim)
                    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(save_to + "_" + str(int(cap.get(cv2.CAP_PROP_POS_FRAMES))) + ".jpg", roi)

                    count += 1

        except Exception as e:
            print(e)
        
        if cv2.waitKey(25) == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()


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

    print("extracting faces from target video.")
    #face_extraction(class_=target_label, video_file=target_video, face_dim=(128, 128), save_to="datasets/images/" + str(target_label), show=False)
    print("extracting faces from source video.")
    face_extraction(class_=source_label, video_file=source_video, face_dim=(128, 128), save_to="datasets/images/" + str(source_label), show=False)

    
