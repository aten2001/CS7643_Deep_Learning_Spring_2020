import csv
import cv2
import numpy as np
lines = []
with open('./training_data/driving_log.csv') as log_file:
	reader = csv.reader(log_file)
	for line in reader:
		lines.append(line)

images_path = []
labels = []
for line in lines:
	threshold = 0.08
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		#image = cv2.imread(current_path)
		flip_probability = np.random.random()
		label = float(line[3])
		if np.abs(label) > threshold or flip_probability > 0.7:
			images_path.append(filename)
			labels.append(label)

def extractFeature(path, label):
	flip_probability = np.random.random()
	current_path = './training_data/IMG/' + path
	image = cv2.imread(current_path)
	if flip_probability > 0.3:
		return (image, label)
	else:
		return (cv2.flip(image, 1), label*-1.0)

def generator(features_path, labels, batch_size, input_shape):
	batch_features = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
	batch_labels = np.zeros((batch_size, 1))
	while True:
		for i in range(batch_size):
			index = np.random.choice(len(features_path), 1)[0]
			batch_features[i], batch_labels[i] = extractFeature(features_path[index], labels[index])
		yield (batch_features, batch_labels)

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

model = Sequential()
model.add(Lambda(lambda x : x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25), (0,0))))
model.add(Convolution2D(24,5,5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(36,5,5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(48,3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

train_feature_path, valid_feature_path, train_labels, valid_labels = train_test_split(images_path, labels, test_size=0.2, random_state=0)
train_generator = generator(train_feature_path, train_labels, batch_size = 32, input_shape = (160,320,3))
valid_generator = generator(valid_feature_path, valid_labels, batch_size = 32, input_shape = (160,320,3))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_feature_path), validation_data=valid_generator, \
	nb_val_samples=len(valid_feature_path), nb_epoch=2)

model.save('model.h5')



