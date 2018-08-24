import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

my_faces_path='./my_faces'
size = 64
name=[]
for n in os.listdir(my_faces_path):
    name.append(n)
labels=np.array(name) 

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, labels.shape[0]+1])

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
    # 第一层
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([8*16*32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,labels.shape[0]+1])
    bout = biasVariable([labels.shape[0]+1])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

output = cnnLayer()  
predict = tf.argmax(output, 1)  
   
saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, tf.train.latest_checkpoint('./model'))  
   
def is_my_face(image):  
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})  
     
    return res[0]

haar = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
haar1 = cv2.CascadeClassifier('./haarcascade_profileface.xml')

cam = cv2.VideoCapture(1)
while (cam.isOpened()):  
    _, img = cam.read()  
    right_faces=cv2.flip(img,1,dst=None)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray_img, 1.3, 5)
    part_faces=haar1.detectMultiScale(gray_img, 1.3, 5)
    part_faces1=haar1.detectMultiScale(right_faces, 1.3, 5)
    if not len(faces):
        print('Can`t get face.')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break      
    for f_x, f_y, f_w, f_h in faces:
        face = img[f_y:f_y+f_h, f_x:f_x+f_w]
        # 调整图片的尺寸
        face = cv2.resize(face, (size,size))
        print('Is this my face? %s' % is_my_face(face))
       
        for j in range(labels.shape[0]):
            if is_my_face(face)==j:
                
                cv2.putText(img, 'this is %s'% name[j], (f_x, f_y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
    for f_x, f_y, f_w, f_h in  part_faces:
        face = img[f_y:f_y+f_h, f_x:f_x+f_w]
              
        face = cv2.resize(face, (IMGSIZE, IMGSIZE))
              
        face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
              
        cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)

        cv2.putText(img, 'you', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2) 
        img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
        n+=1
        for j in range(labels.shape[0]):
            if is_my_face(face)==j:
                
                cv2.putText(img, 'this is %s'% name[j], (f_x, f_y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
    for f_x, f_y, f_w, f_h in  part_faces1:
        face = img[f_y:f_y+f_h, f_x:f_x+f_w]
              
        face = cv2.resize(face, (IMGSIZE, IMGSIZE))
              
        face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
              
        cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)

        cv2.putText(img, 'you', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2) 
        img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
        n+=1
    cv2.imshow('image',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
cam.release()
cv2.destroyAllWindows()
sess.close() 
sys.exit(0)