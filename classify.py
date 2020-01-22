#Importing Libraries
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

#Constructuing the Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help = "path to model")
ap.add_argument("-l", "--labelbin", required = True, help = "path to label binarizer file")
ap.add_argument("-i", "--image", required = True, help = "path to input image")
args = vars(ap.parse_args())

#Loading the image
image = cv2.imread(args["image"])
output = image.copy()

#Preprocessing the image
image = cv2.resize(image, (96, 96))
image = image.astype("float")/255.0
image = img_to_array(image)
image = np.expand_dims(image, axis = 0)

#loading the trained convolutional neural network and the label binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

#Classifying image
print("[INFO] classifying image...")
pred = model.predict(image)
proba = pred[0]

	
idx = np.argmax(proba)
label = lb.classes_[idx]	

if label == 0:
	l = "French Fries"

elif label == 1:
	l = "Ice Cream"
			
elif label == 2:
	l = "Pizza"
			
elif label == 3:
	l = "Samosa"
			
elif label == 4:
	l = "Waffle"
		
os.system("clear")
print("Food is -  {}".format(l))
print("Accuracy is {:.2f}".format(100*proba[idx])) 
