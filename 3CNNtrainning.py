import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
import dlib
from sklearn.model_selection import train_test_split
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpuConfig.gpu_options.allow_growth = True  

size=64
imgs = []
labs = []

input_dir= './my_faces'
def my_faces(input_dir):
    for (path, dirnames, filenames) in os.walk(input_dir):
        for dirname in dirnames:
                img_path = path+'/'+dirname
                return img_path
my_faces_path= my_faces(input_dir)
other_faces_path = './other_faces'

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(path , h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top,bottom,left,right = getPaddingSize(img)
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)
            

readData(my_faces_path)
readData(other_faces_path)

imgs = np.array(imgs)
labs = np.array([[0,1] if lab == my_faces_path else [1,0] for lab in labs])

train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.25, random_state=random.randint(0,100))

train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0
print('train size:%s, test size:%s' % (len(train_x), len(test_x)))

batch_size = 50
num_batch = len(train_x) // batch_size

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
   
    W1 = weightVariable([3,3,3,32]) 
    b1 = biasVariable([32])
   
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
   
    pool1 = maxPool(conv1)
   
    drop1 = dropout(pool1, keep_prob_5)

  
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

 
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

   
    Wf = weightVariable([8*8*64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

 
    Wout = weightVariable([512,2])
    bout = weightVariable([2])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

def cnnTrain():
    since = time.time()
    
    out = cnnLayer()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
  
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()

    predict = tf.argmax(out, 1)  
    saver = tf.train.Saver()

    with tf.Session(config=gpuConfig) as sess:
        
        sess.run(tf.global_variables_initializer())
        
        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        for i in range(num_batch):
            batch_x = train_x[i*batch_size : (i+1)*batch_size]
            batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                
            _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
            summary_writer.add_summary(summary, num_batch+i)
                # 打印损失
            print(num_batch+i, loss)

            if i == num_batch-2:
                # 获取测试数据的准确率
                acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                print('accuracy')
                print(num_batch+i, acc)
                saver.save(sess, './train_faces.model', global_step=num_batch+i)
                print ('saver done')
                time_elapsed = time.time() - since
                
                print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
                sys.exit(0)


cnnTrain()