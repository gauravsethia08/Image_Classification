#IMporting Libraries
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as k

class SmallerVGNet:
	@staticmethod
	def build(width, height, depth, classes):
		#Initializing the model along with the input shape to be "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		
		#Updating Input shape if we using "channels first"
		if k.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1
			
		#Conv => ReLU => Pool
		model.add(Conv2D(32, (3,3), padding = "same", input_shape = inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = chanDim))
		model.add(MaxPooling2D(pool_size = (3,3)))
		model.add(Dropout(0.25))
		
		#(Conv => ReLU)*2
		model.add(Conv2D(64, (3,3), padding = "same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = chanDim))
		model.add(Conv2D(64, (3,3), padding = "same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = chanDim))
		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(Dropout(0.25))
		
		#(Conv => ReLU)*2
		model.add(Conv2D(128, (3,3), padding = "same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = chanDim))
		model.add(Conv2D(128, (3,3), padding = "same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis = chanDim))
		model.add(MaxPooling2D(pool_size = (2,2)))
		model.add(Dropout(0.25))
		
		#first (and only) set for FC => ReLU layers
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		
		#softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		
		return model
