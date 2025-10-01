import numpy as np
import cv2 as cv
import torch
from ultralytics import YOLO
from PIL import Image
import pyttsx3
import schedule
import os

# os.system("/usr/bin/espeak-ng ' '")

engine = pyttsx3.init(driverName='espeak')

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

    # announcement = f'I see a {identified_objects[0]}'
    # engine.say(announcement)
    # engine.runAndWait()

schedule.every(3).seconds.do(textToSpeech)
    

FRAME_SKIP = 5 
frame_counter = 0

model = YOLO("yolov5x.pt")
model.cuda()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    schedule.run_pending()

    # --- Display the live video stream ---
    # cv.imshow('Live Frame', frame)
    
    if cv.waitKey(1) == ord('q'):
        break
        
    # --- Model Processing Logic (every N frames) ---
    if frame_counter % FRAME_SKIP == 0:

        results = model(frame, verbose=False)
        identified_objects = [] 
        
        for result in results:
            identified_objects = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        
        
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv.imshow("YOLO Results", annotated_frame)
    
    frame_counter += 1
    

cap.release()
cv.destroyAllWindows()