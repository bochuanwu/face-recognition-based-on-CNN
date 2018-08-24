import os
import glob
import tensorflow as tf 
from tensorflow.contrib.layers import *
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
import numpy as np
from random import shuffle
 
age_table=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
sex_table=['female','male']  # female:女; male:男
 
# AGE==True 训练年龄模型，False,训练性别模型
AGE = False
 
if AGE == True:
	lables_size = len(age_table) # 年龄
else:
	lables_size = len(sex_table) # 性别
 
face_set_fold = './AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification'
 
fold_0_data = os.path.join(face_set_fold, 'fold_0_data.txt')
fold_1_data = os.path.join(face_set_fold, 'fold_1_data.txt')
fold_2_data = os.path.join(face_set_fold, 'fold_2_data.txt')
fold_3_data = os.path.join(face_set_fold, 'fold_3_data.txt')
fold_4_data = os.path.join(face_set_fold, 'fold_4_data.txt')
 
face_image_set = os.path.join(face_set_fold, 'aligned')
 
 
batch_size = 1
 
# 缩放图像的大小
IMAGE_HEIGHT = 227
IMAGE_WIDTH = 227
# 读取缩放图像
jpg_data = tf.placeholder(dtype=tf.string)
decode_jpg = tf.image.decode_jpeg(jpg_data, channels=3)
resize = tf.image.resize_images(decode_jpg, [IMAGE_HEIGHT, IMAGE_WIDTH])
resize = tf.cast(resize, tf.uint8) / 255
def resize_image(file_name):
	with tf.gfile.FastGFile(file_name, 'rb') as f:
		image_data = f.read()
	with tf.Session() as sess:
		image = sess.run(resize, feed_dict={jpg_data: image_data})
	return image
 
 
X = tf.placeholder(dtype=tf.float32, shape=[batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
Y = tf.placeholder(dtype=tf.int32, shape=[batch_size])
 
def conv_net(nlabels, images, pkeep=0.75):
	weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
	with tf.variable_scope("conv_net", "conv_net", [images]) as scope:
		with tf.contrib.slim.arg_scope([convolution2d, fully_connected], weights_regularizer=weights_regularizer, biases_initializer=tf.constant_initializer(1.), weights_initializer=tf.random_normal_initializer(stddev=0.005), trainable=True):
			with tf.contrib.slim.arg_scope([convolution2d], weights_initializer=tf.random_normal_initializer(stddev=0.01)):
				conv1 = convolution2d(images, 96, [7,7], [4, 4], padding='VALID', biases_initializer=tf.constant_initializer(0.), scope='conv1')
				pool1 = max_pool2d(conv1, 3, 2, padding='VALID', scope='pool1')
				norm1 = tf.nn.local_response_normalization(pool1, 5, alpha=0.0001, beta=0.75, name='norm1')
				conv2 = convolution2d(norm1, 256, [5, 5], [1, 1], padding='SAME', scope='conv2') 
				pool2 = max_pool2d(conv2, 3, 2, padding='VALID', scope='pool2')
				norm2 = tf.nn.local_response_normalization(pool2, 5, alpha=0.0001, beta=0.75, name='norm2')
				conv3 = convolution2d(norm2, 384, [3, 3], [1, 1], biases_initializer=tf.constant_initializer(0.), padding='SAME', scope='conv3')
				pool3 = max_pool2d(conv3, 3, 2, padding='VALID', scope='pool3')
				flat = tf.reshape(pool3, [-1, 384*6*6], name='reshape')
				full1 = fully_connected(flat, 512, scope='full1')
				drop1 = tf.nn.dropout(full1, pkeep, name='drop1')
				full2 = fully_connected(drop1, 512, scope='full2')
				drop2 = tf.nn.dropout(full2, pkeep, name='drop2')
	with tf.variable_scope('output') as scope:
		weights = tf.Variable(tf.random_normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')
		biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
		output = tf.add(tf.matmul(drop2, weights), biases, name=scope.name)
	return output

def detect_age_or_sex(image_path):
	logits = conv_net(lables_size, X)
	saver = tf.train.Saver()
 
	with tf.Session() as sess:
		saver.restore(sess, './age.module' if AGE == True else './sex.module')
		
		softmax_output = tf.nn.softmax(logits)
		res = sess.run(softmax_output, feed_dict={X:[resize_image(image_path)]})
		res = np.argmax(res)
 
		if AGE == True:
			return age_table[res]
		else:
			return sex_table[res]

sex=detect_age_or_sex('./test.jpg')
print(sex)