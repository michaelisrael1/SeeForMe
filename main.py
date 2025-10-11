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
# os.system("/usr/bin/espeak-ng ' '")

engine = pyttsx3.init(driverName='espeak')
client = boto3.client('rekognition')


global identified_objects
identified_objects = []


def textToSpeech():
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


schedule.every(.5).seconds.do(textToSpeech)


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

    schedule.run_pending()

    # --- Display the live video stream ---
    cv.imshow('Live Frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

    # --- Model Processing Logic (every N frames) ---
    # if frame_counter % FRAME_SKIP == 0:
    if test == 10:
        identified_objects = []

        success, buffer = cv.imencode('.jpg', frame)
        print(f'Converting frame to jpg')
        if success:
            img = buffer.tobytes()
            image_obj = RekognitionImage(
                {'Bytes': img}, f"Frame {test}", client)
            labels = image_obj.detect_labels(max_labels=10, min_confidence=55)
            for label in labels:
                identified_objects.append(label['Name'])
            textToSpeech()
    # Display the annotated frame
    # cv.imshow("YOLO Results", annotated_frame)
    test += 1
    frame_counter += 1


# cap.release()
cv.destroyAllWindows()
