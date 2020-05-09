import cv2
import dlib
import pickle
import numpy as np
import os.path


import glob
import random
import math
import itertools
from sklearn.svm import SVC
import pickle

from cv2 import WINDOW_NORMAL
from sample import get_landmarks
from sample import data

emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"] 
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
ESC = 27
def start_webcam_linear(model_emotion, window_size, window_name='live', update_time=50):
    training_data = []
    file = open("results.txt", "w")
    da = [[]]
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    video_feed = cv2.VideoCapture(0)
    video_feed.set(3, width)
    video_feed.set(4, height)
    read_value, webcam_image = video_feed.read()

    delay = 0
    init = True
    while read_value:
        read_value, webcam_image = video_feed.read()
        gray = cv2.cvtColor(webcam_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        detections = detector(webcam_image, 0)
        da = get_landmarks(clahe_image)
        if da == "error":
            print("no face detected on this one")
        else:
            training_data.append(da)
        #append image array to training data list
        for k,d in enumerate(detections): #For all detected face instances individually
            emotion_prediction = model_emotion.predict(training_data)
            emotion_probability = model_emotion.predict_proba(training_data)
            print(emotion_probability)
            print(d.left(), d.top(), d.right(), d.bottom())
            cv2.rectangle(webcam_image,(d.left(), d.top()),(d.right(), d.bottom()),(0,0,255),3)
            cv2.putText(webcam_image, emotions[emotion_prediction[0]], (d.left(), d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            for i in range(6):
                for j in range(1):
                    file.write(emotions[i])
                    file.write(":")
                    file.write(str(emotion_probability[j][i]))
                    file.write("\n")
        delay += 1
        delay %= 20
        cv2.imshow(window_name, webcam_image)
        key = cv2.waitKey(update_time)
        if key == ESC:
            break

    file.close()
    cv2.destroyWindow(window_name)


def analyze_picture_linear(model_emotion, path, window_size, window_name='static'):
    training_data = []
    f = open("result_image.txt", "w")
    f.truncate(0)
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)
    image = cv2.imread(path, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1)
    da = get_landmarks(clahe_image) 
    if da == "error":
        print("no face detected on this one")
    else:
        training_data.append(da) #append image array to training data list
        print(training_data)
    for k,d in enumerate(detections): #For all detected face instances individually
        emotion_prediction = model_emotion.predict(training_data)
        print(emotion_prediction[0])
        emotion_probability = model_emotion.predict_proba(training_data)
        print(emotion_probability)
        print(d.left(), d.top(), d.right(), d.bottom())
        cv2.rectangle(image,(d.left(), d.top()),(d.right(), d.bottom()),(0,0,255),3)
        index_max = np.argmax(emotion_probability[0])
        print(index_max)
        cv2.putText(image, emotions[index_max], (d.left(), d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        for i in range(6):
                for j in range(1):
                    f.write(emotions[i])
                    f.write(": ")
                    f.write(str(emotion_probability[j][i]))
                    f.write("\n")
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0)
    if key == ESC:
        cv2.destroyWindow(window_name)
    f.close()
    return image

if __name__ == '__main__':
    emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"] 
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    run_loop = True
    window_name = "Facifier Static (press ESC to exit)"
    print("Default path is set to data/sample/")
    print("Type q or quit to end program")
    choice = input("Use webcam?(y/n) ")
    if (choice == 'y'):
        window_name = "Facifier Webcam (press ESC to exit)"
        start_webcam_linear(loaded_model, window_size=(1200, 720), window_name=window_name, update_time=15)
    elif (choice == 'n'):
        run_loop = True
        window_name = "Facifier Static (press ESC to exit)"
        print("Default path is set to data/sample/")
        print("Type q or quit to end program")
        while run_loop:
            path = "../data/sample/"
            file_name = input("Specify image file: ")
            if file_name == "q" or file_name == "quit":
                run_loop = False
            else:
                path += file_name
                if os.path.isfile(path):
                    analyze_picture_linear(loaded_model, path, window_size=(1280, 720), window_name=window_name)
                else:
                    print("File not found!")
    else:
        print("Invalid input, exiting program.")
    
