import numpy as np
import cv2
import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Video Test')
parser.add_argument('--video', help='Path to video file.',required=True)
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('filename.mp4', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         60, size)

def binarization(img):
    ret1,th1 = cv2.threshold(img,180,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(th1,kernel,iterations = 2)
    return dilation

while cv2.waitKey(1) < 0:
    # get frame from the video
    hasFrame, image = cap.read()
    if not hasFrame:
        print("Done processing !!!")

        cap.release()
        break
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = binarization(gray)

    contours,_ = cv2.findContours(gray, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) != 0:
        #print(0)
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)


        width = (x + w)
        height = (y + h)
        #print(w)
        cv2.rectangle(orig,(x,y),(width,height),(225, 0, 0),2)
    cv2.imshow("video", orig)
    result.write(orig)
cap.release()
result.release()

cv2.destroyAllWindows()