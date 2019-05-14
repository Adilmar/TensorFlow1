import tensorflow as tf 
from tensorflow import keras
import os
import pandas
import numpy as np

def treinar():
	#define model neural network 
	model = keras.Sequential()
	#input layer - use tanh activation because job numbers 1 between -1
	input_layer = keras.layers.Dense(3, input_shape=[3], activation='tanh')
	model.add(input_layer)

	#output layer use sigmoid because my output 0 or 1.
	output_layer = keras.layers.Dense(1, activation='sigmoid')
	model.add(output_layer)

	# gradient of otimization train 
	gd = tf.train.GradientDescentOptimizer(0.01)

	model.compile(optimizer=gd, loss='mse')
    
        #open data-files
	file = ("csv/play_tennis.csv")

	def data_encode(file):
	    X = []
	    Y = []
	    train_file = open(file, 'r')
	    for line in train_file.read().strip().split('\n'):
	        line = line.split(',')
	        X.append([int(line[0]), int(line[1]), int(line[2])])
	        Y.append(int(line[3]))
	    return X, Y

	train_X , train_Y = data_encode(file)

	training_x = np.array(train_X)
	training_y = np.array(train_Y)

	model.fit(training_x, training_y, epochs=8000, steps_per_epoch=10)
	model.save('tennis.h5')

	#model.load_weights('tennis.h5')
	text_x = np.array([[1, 0, 0],[1,1,0]])
	prediction = model.predict(text_x, verbose=0, steps=1)
	print("predicao",prediction)


def predicao(entrada):
	#load complete model 
	model = keras.models.load_model('tennis.h5')
	
	#prin model resume 
	#model.summary()

	#create input prediction
	text_x = np.array(entrada)
	
	#calcule prediction 
	prediction = model.predict(text_x, verbose=0, steps=1)
	print("predicao",prediction)
	

