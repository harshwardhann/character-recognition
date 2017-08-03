import numpy as np
from scipy.io import loadmat
import csv
import tensorflow as tf
import random


def loaddata(training,testing):
	training_data = loadmat(training)
	testing_data = loadmat(testing)
	xTr = training_data["x"]
	yTr = np.round(training_data["y"])
	xTe = testing_data["x"]
	return xTr,yTr,xTe

def train(xTr,yTr):
	sess = tf.InteractiveSession()

	x = tf.placeholder(tf.float32, shape=[ 784,None])
	y_ = tf.placeholder(tf.float32, shape=[10, None])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	sess.run(tf.global_variables_initializer())
	y = y = tf.matmul(x,W) + b
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	for i in range(4000):
  		train_step.run(feed_dict={x: xTr[i], y_: yTr[i]})

  	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(accuracy)
	#print(sess.run(accuracy, feed_dict={x: xTe, y_: mnist.test.labels}))
	


def test(xTe,w):
	pass

	return 0

if __name__ == "__main__":
	xTr,yTr,xTe = loaddata("train.mat","test.mat") 
	train(xTr,yTr)

	writes results to result.csv
	with open("results.csv", 'w') as csvfile:
		writer = csv.writer(csvfile)
	 	writer.writerow(['id','digit'])
	 	for i in range(n):
	 		pred = pred[i]
	 		x = 0
	 		for j in range(10):
	 			if pred[j]==1:
	 				writer.writerow([i, j])

