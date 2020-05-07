 
import cv2
import os
import numpy as np

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

from sample import get_landmarks
from sample import data
emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"] 
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
# Load images
def analyze_v2(path):
	emotions = ["afraid", "angry", "disgusted", "happy", "neutral", "sad", "surprised"] 
	model_emotion = pickle.load(open('finalized_model.sav', 'rb'))
	training_data = []
	f = open("result_image.txt", "w")
	f.truncate(0)
	folder='uploads'
	#image = cv2.imread(os.path.join(folder, path))
	image = cv2.imread(os.path.join(folder, path), 1)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image = clahe.apply(gray)
	detections = detector(clahe_image, 1)
	da = get_landmarks(clahe_image) 
	if da == "error":
		print("no face detected on this one")
	else:
		training_data.append(da) #append image array to training data list
		
	for k,d in enumerate(detections): #For all detected face instances individually
		emotion_prediction = model_emotion.predict(training_data)
		emotion_probability = model_emotion.predict_proba(training_data)
		print(d.left(), d.top(), d.right(), d.bottom())
		cv2.rectangle(image,(d.left(), d.top()),(d.right(), d.bottom()),(0,0,255),3)
		cv2.putText(image, emotions[emotion_prediction[0]], (d.left(), d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		for i in range(6):
				for j in range(1):
					f.write(emotions[i])
					f.write(": ")
					f.write(str(emotion_probability[j][i]))
					f.write("\n")
	cv2.imwrite(os.path.join(folder,path),image)
	f.close()
	return path

def analyze_picture_linear(path):
	loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
	training_data = []
	da = [[]]
	f = open("result_image.txt", "w")
	f.truncate(0)
	folder='uploads'
	image = cv2.imread(os.path.join(folder, path))
	r = 980.0 / image.shape[1]
	dim = (980, int(image.shape[0] * r))
	
	# perform the actual resizing of the image and show it
	image= cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	clahe_image = clahe.apply(gray)
	detections = detector(clahe_image, 1)
	da = get_landmarks(clahe_image) 
	if da == "error":
		print("no face detected on this one")
	else:
		training_data.append(da) #append image array to training data list
		
	for k,d in enumerate(detections): #For all detected face instances individually
		emotion_prediction = loaded_model.predict(training_data)
		emotion_probability = loaded_model.predict_proba(training_data)
		print(d.left(), d.top(), d.right(), d.bottom())
		cv2.rectangle(image,(d.left(), d.top()),(d.right(), d.bottom()),(0,0,255),3)
		cv2.putText(image, emotions[emotion_prediction[0]], (d.left(), d.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		for i in range(6):
				for j in range(1):
					f.write(emotions[i])
					f.write(": ")
					f.write(str(emotion_probability[j][i]))
					f.write("\n")

	cv2.imwrite(os.path.join(folder,path),image)
	f.close()
	return path 

analyze_picture_linear('D:\\New folder\\facultate\\Emotion-Recognition\\src\\opencv-flask\\uploads\\8.png')