#Setting Matplotlib backend, so the figures are saved in background
import matplotlib
matplotlib.use("Agg")

#Importing Libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from include.smallervggnet import SmallerVGNet
from sklearn import preprocessing
from imutils import paths
import numpy as np
import argparse
import random 
import pickle
import cv2
import os

#Constructing the Argument Parse
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to dataset")
ap.add_argument("-m", "--model", required = True, help = "path to output model")
ap.add_argument("-l", "--labelbin", required = True, help = "path to output label binarizer")
ap.add_argument("-p", "--plot", default = "plot.png" , help = "path to output plot")
args = vars(ap.parse_args())

#Initializing the number of epoches, learning rate, batch size and image dimension
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

#Initilaize data and label 
data = []
labels = []

#get the image path and randomly shuffle them
print("[INFO] loading images ...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

#Looping over inout images and preprocessing it
for imagePath in imagePaths:
	
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)
	
	#Extracting labels
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
	
#Scaling the RAW pixel intensity to range between [0,1]
data = np.array(data, dtype = "float")/255.0
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)
labels = np.array(labels)
print(labels)
print("[INFO] data matrix : {:.2f}MB".format(data.nbytes/(1024*1000.0)))

#Binarize the labels
lb = LabelBinarizer()
lbales = lb.fit_transform(labels)

#Train Test Split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.2, random_state = 42)

print(trainX)
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

#Initliase the model
print("[INFO] compliing model...")
model = SmallerVGNet.build(width = IMAGE_DIMS[1], height = IMAGE_DIMS[0], depth = IMAGE_DIMS[2], classes = len(lb.classes_))
opt = Adam(lr = INIT_LR, decay = INIT_LR/EPOCHS)
model.compile(loss = "sparse_categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

#Training the network
print("[INFO] training the network...")
H = model.fit_generator(aug.flow(trainX, trainY, BS), validation_data = (testX, testY), steps_per_epoch = len(trainX)//BS, epochs=EPOCHS, verbose = 1)

#Saving model to disk
print("[INFO] seralising network ...")
model.save(args["model"])

#Saving the label binarizer
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

#Plotting the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
