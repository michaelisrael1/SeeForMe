from numpy import ndarray
import cv2 as cv
import torch
from ultralytics import YOLO
from PIL import Image
import pyttsx3
import schedule
import os
import boto3
import base64
from image import RekognitionImage
from pprint import pprint
import requests
import json
# os.system("/usr/bin/espeak-ng ' '")

engine = pyttsx3.init(driverName='espeak')
client = boto3.client('rekognition')

SERVER_URL = "http://127.0.0.1:5000/detect/"


def textToSpeech(identified_objects):
    if len(identified_objects) == 0:
        return
    print(identified_objects)

    identified = {}

    for seen in identified_objects:
        if seen in identified:
            identified[seen] += 1
            continue
        identified[seen] = 1

    for key, value in identified.items():
        announcement = f'I see {value} {key}'
        engine.say(announcement)
        engine.runAndWait()


def send_to_AWS(img_bytes):
    identified_objects = []
    image_obj = RekognitionImage(
        {'Bytes': img_bytes}, f"Frame {test}", client)
    labels = image_obj.detect_labels(max_labels=10, min_confidence=55)
    for label in labels:
        identified_objects.append(label['Name'])
    return identified_objects


def send_to_local_api(img_bytes):
    files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
    response = requests.post(SERVER_URL, files=files, timeout=30)
    pprint(response.json())
    return


def get_encode_image(frame):
    success, buffer = cv.imencode('.jpg', frame)
    print('Converting frame to jpg')
    if success:
        return buffer.tobytes()


FRAME_SKIP = 60
frame_counter = 0

# model = YOLO("yolov5x.pt")
# model.cuda()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' in the video window to quit.")

test = 0

while test <= 10:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # --- Display the live video stream ---
    cv.imshow('Live Frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

    # --- Model Processing Logic (every N frames) ---
    # if frame_counter % FRAME_SKIP == 0:
    if test == 10:
        # labels = send_to_AWS(frame)
        img_bytes = get_encode_image(frame)
        # detected = send_to_AWS(img_bytes)
        # textToSpeech(detected)
        send_to_local_api(img_bytes)
    # Display the annotated frame
    # cv.imshow("YOLO Results", annotated_frame)
    test += 1
    frame_counter += 1


# cap.release()
cv.destroyAllWindows()
