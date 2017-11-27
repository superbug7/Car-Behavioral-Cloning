import csv
import cv2
import numpy as np
import sklearn

from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split

lines = []

with open('./driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(lines, batch_size=32):
	num_lines = len(lines)
	while 1:
		#shuffle(lines)		
		for offset in range(0,num_lines, batch_size):
			batch_samples = lines[offset:offset+batch_size]		

			images = []
			measurements = []
			for batch_sample in batch_samples:
				source_path = batch_sample[0]
				filename = source_path.split('/')[-1]
				current_path = './IMG/' + filename
				print (current_path)
				image = cv2.imread(current_path)
				images.append(image)
				measurement = float(batch_sample[3])
				measurements.append(measurement)

			augmented_images, augmented_measurements = [], []
			for image,measurement in zip(images, measurements):
				augmented_images.append(image)
				augmented_measurements.append(measurement)
				augmented_images.append(cv2.flip(image,1))
				augmented_measurements.append(measurement*-1.0)

			X_train = np.array(augmented_images)
			y_train = np.array(augmented_measurements)
			print( X_train.shape)
			print(y_train.shape)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 80, 320

model = Sequential()
model.add(Lambda(lambda x: ( x / 255.0 ) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
#model.add(Flatten())
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
#model.add(MaxPooling2D())
#model.add(Dropout(0.25))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=5)
model.save('model.h5')

