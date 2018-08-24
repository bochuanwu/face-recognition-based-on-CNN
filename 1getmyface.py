import os
import random
import numpy as np
import cv2

def createdir(*args):
 
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

IMGSIZE = 64



def relight(imgsrc, alpha=1, bias=0):
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc


def getfacefromcamera(outdir):
    createdir(outdir)
    camera = cv2.VideoCapture(1)
  
    haar = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
    n = 1
    while 1:
        if (n <= 1000):
            print('It`s processing %s image.' % n)
           
            success, img = camera.read()
           
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
            for f_x, f_y, f_w, f_h in faces:
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
              
                face = cv2.resize(face, (IMGSIZE, IMGSIZE))
              
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
              
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)

                cv2.putText(img, 'you', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2) 
                img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                n+=1
            cv2.imshow('img', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    name = input('please input yourename: ')
    getfacefromcamera(os.path.join('./my_faces', name))