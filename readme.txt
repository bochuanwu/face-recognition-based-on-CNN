人脸识别

这个程序首先基于opencv获取摄像头录入的可识别数据集，包含标签（label）以及数据（data）。 其次通过开源人脸数据库（LFW）下载一定量的非识别用数据集。所有数据都需要进行人脸检测然后进行裁剪，已获取最佳识别效果。
在将所有数据预处理好后，建立一个基于tensorflow的卷积神经网络模型。本程序使用了三层神经网络，每层神经网络都由卷积层，池化层（使用maxpooling），dropout层组成，之后由一个全连接层，输出标签。损失函数参考于facenet的Triplet loss三重损失函数。
训练好模型之后可以随时调用模型识别新的图像并得到识别结果。
实际运行时间主要在于模型训练，根据训练集大小不同，所需训练时间不同，10k数据的训练所需时间约为：暂未获取
this project is based on openCV and tensorflow
the process of using this code is following

1getmyface- 先输入文件名 然后通过摄像头获取人脸(get your face by input a filename-if you have a vedio camera)
2trainingothers- 预处理开源人脸库的数据( preprocessing the dataset from lfw(if you already have some dataset,ignore this)
3CNNtrainning/3multiplepersonCNNtrainning- 单标签/多标签模型训练（根据第一步获取一个文件夹还是多个文件夹）(you can train model for one person or many people）
4checkwhoami/4multiplecheck- 单人判别（single person recognition）/多人判别（multiple person recognition）（check by vedio camera！）

update 2018/8/9
添加了年龄及性别检测
added age and sex recognition
update 2018/8/14
添加了多角度人脸识别
added multiple view face detection（using 1multipleview.py to get face and 3 to train model,then use 4.multpeveiwcheck.py to do face recognition）


by bochuan Wu-https://github.com/bochuanwu
