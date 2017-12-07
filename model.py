import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
import math
#%matplotlib inline

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D
from keras.layers import Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam


lines = []
center = []
left = []
right = []
measurements = []


def flip_image(image, measurement):
  image = cv2.flip(image,1)
  measurement = measurement*(-1)
  return image, measurement



with open('./driving_log.csv') as csvFile:
	reader = csv.reader(csvFile)
	#if skipHeader:
	#	next(reader, None)
	for line in reader:
		lines.append(line)
	#return lines

for line in lines:
	center.append('IMG/' + line[0].split('/')[-1])
	left.append('IMG/' + line[1].split('/')[-1])
	right.append('IMG/' + line[2].split('/')[-1])
	measurements.append(np.float32(line[3].split('/')[-1]))

#for i in center:
#	new_img, new_m = flip_image(cv2.imread(center[i]))
#	center.append
#print(center[0])
#center_size = (len(center)) 
#(len(left)) 
#nprint(len(right)) 
#print(len(measurements)) 

center, measurements = shuffle(center, measurements)
center, x_val, measurements, y_val = train_test_split(center, measurements, test_size = 0.10, random_state = 100) 	

#print(measurements) 
img_c=[]
img_l=[]
img_r=[]
angle_c=[]
angle_l=[]
angle_r=[]

for m in measurements:
	#print(m)
	index = measurements.index(m)
	if m > 0.15:
		img_r.append(center[index])
		angle_r.append(m)
	if m < -0.15:
		img_l.append(center[index])
		angle_l.append(m)
	else:
		img_c.append(center[index])
		angle_c.append(m)


samples_center_left = len(img_c) - len(img_l)
samples_center_right = len(img_c) - len(img_r)
indexes_L = random.sample(range(len(center)), samples_center_left)
indexes_R = random.sample(range(len(center)), samples_center_right)


for index in indexes_L:
	if measurements[index] < -0.15:
		img_l.append(right[index])
		angle_l.append(measurements[index] - 0.25)



for index in indexes_R:
	if measurements[index] > 0.15:
		img_r.append(left[index])
		angle_r.append(measurements[index] + 0.25)



x_train = img_r + img_l + img_c
y_train = np.float32(angle_r + angle_l + angle_c )

x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)
#print(x_train[0])
#print(y_train.shape)
#print(x_val.shape)
#print(y_val.shape)
samples, measurements = shuffle(x_train, y_train)
img = cv2.imread(samples[0])
#print(samples[0])

def preprocess_image(img):
    new_img = img[60:140,:,:]
    #plt.imshow(new_img)
    # apply subtle blur
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV)
    new_img = np.array(new_img, dtype = np.float64)
    #Generate new random brightness
    random_bright = .5+random.uniform(0.3,1.0)
    new_img[:,:,2] = random_bright*new_img[:,:,2]
    new_img[:,:,2][new_img[:,:,2]>255]  = 255
    new_img = np.array(new_img, dtype = np.uint8)
    #Convert back to RGB colorspace
    new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
    # scale to 66x200x3 (Nvidia NN wants image in this format)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    #Convert to YUV color space
    new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2YUV)
    return new_img

#samples, measurements = shuffle(x_train, y_train)
#batch_sample = int(np.random.choice(len(samples),1))
#batch_train = preprocess_image(cv2.imread(samples[batch_sample]))
#batch_angle = measurements[batch_sample]*(1+ np.random.uniform(-0.10,0.10))
#print(batch_train.shape)
#print(batch_angle)

def generator_train(batch_size):
      	#num_samples = len(samples)
        batch_train = np.zeros((batch_size,66,200,3), dtype=np.float32)
        batch_angle = np.zeros((batch_size,), dtype=np.float32)
        while 1:
                samples, measurements = shuffle(x_train, y_train)#shuffle(lines)
                for offset in range(batch_size):
                   batch_sample = int(np.random.choice(len(samples),1))
                   batch_train[offset] = preprocess_image(cv2.imread(samples[batch_sample]))
                   batch_angle[offset] = measurements[batch_sample]*(1+ np.random.uniform(-0.10,0.10))
                   if abs(measurements[batch_sample]) > 0.33:
                       new_img, new_m = flip_image(batch_train[offset], batch_angle[offset])
                       np.append(batch_train,new_img)
                       np.append(batch_angle,new_m)
                       if len(batch_train) == batch_size:
                          #print(batch_train.shape)
                          #print(batch_angle.shape)
                          yield batch_train, batch_angle
                        

def generator_val(samples, measurements, batch_size):
        batch_train =  np.zeros((batch_size,66,200,3), dtype=np.float32)
        batch_angle = np.zeros((batch_size,), dtype=np.float32)
        while 1:
           samples, measurements = shuffle(samples, measurements) 
           for offset in range(batch_size):
              batch_sample = int(np.random.choice(len(samples),1))
              batch_train[offset] = preprocess_image(cv2.imread(samples[batch_sample]))
              batch_angle[offset] = measurements[batch_sample]
           yield batch_train, batch_angle



def main(_):
	train_gen = generator_train(128)
	#print(x_val.shape)
	val_gen = generator_val(x_val, y_val, 128)

	model = Sequential()
	model.add(Lambda(lambda x: ( x / 255.0 ) - 0.5, input_shape=(66,200,3)))
	#model.add(Cropping2D(cropping=((70,25),(0,0))))
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
	model.add(Dense(100,W_regularizer=l2(0.001)))
	model.add(Dense(50,W_regularizer=l2(0.001)))
	model.add(Dense(10,W_regularizer=l2(0.001)))
	model.add(Dense(1))
	
	model.summary()

	adam = Adam(lr = 0.0001)
	model.compile(optimizer= adam, loss='mse')
	print(x_train.shape)
	print(x_val.shape)
	model.fit_generator(train_gen, steps_per_epoch=(math.ceil(len(x_train)/128)-1) , validation_data=val_gen, validation_steps=math.ceil(len(x_val)/128)-1, nb_epoch=20)
	#model.fit_generator(train_gen, samples_per_epoch=(math.ceil(len(x_train)/128)-1) * 128, validation_data=val_gen, nb_val_samples=len(x_val), nb_epoch=15)
	#model.save('model.h5')


if __name__ == '__main__':
	tf.app.run()
