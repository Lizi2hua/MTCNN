from core import detector
from core import old_detector
import os
from PIL import  Image
from NNTools.draw_rectangle import draw_rect
import numpy as np
import cv2
import time
if __name__ == '__main__':
    video_path=r"test_videos/test.mp4"
    # detector = detector.Detector()
    detector = old_detector.Detector()
    video = cv2.VideoCapture(video_path)
    frame_count=0
    start_time = time.time()
    while True:
        ret,frame=video.read()
        frame_count+=1
        if ret:
            b, g, r = cv2.split(frame)
            img = cv2.merge([r, g, b])#opencv用的图片格式是BGR，框架用的是RGB，注意数据分开
            img=Image.fromarray(img)#w,h=img.size这儿会报错，故加这一行
            boxes=detector.detect(img)
            # boxes=opt_detector.detect(im)
            if boxes.shape[0] !=0:
                cv2.rectangle(frame,(boxes[:,1],boxes[:,2]),(boxes[:,3],boxes[:,4]),(0,0,255),thickness=2)
            else:
                pass
            # if boxes
            cv2.imshow("detect_face",frame)
            cv2.waitKey(21)
        else:
            print("videos ended or failed to read!")
            break
    end_time=time.time()
    cost=end_time-start_time
    print('==================================')
    print("avg_FPS:",frame_count/cost)

