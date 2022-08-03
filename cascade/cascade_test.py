# encoding:utf-8
import cv2
import numpy as np
import time
 



def cascade_test(img,clock_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clock = clock_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    # faces = clock_cascade.detectMultiScale(gray)
    print('Detected ', len(clock), " face")
    print(clock)
    for (x, y, w, h) in clock:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow('img', img)
def cameraCap(clock_cascade):
    
    # 读取图像并检测脸部
    cap =cv2.VideoCapture(0)
    while True:
        rat , img = cap.read()
        cascade_test(img,clock_cascade)
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def pictureCap(clock_cascade):
    img = cv2.imread('C://Users/danny/airLab/CVclock/img/149787.jpg')
    img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
    cascade_test(img,clock_cascade)
    cv2.waitKey(0)
if __name__ =='__main__':
    clock_cascade = cv2.CascadeClassifier('cascade2/cascade.xml')
    cameraCap(clock_cascade)