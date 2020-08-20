from core import detector
import os
from PIL import  Image
from NNTools.draw_rectangle import draw_rect
if __name__ == '__main__':
    image_path=r"test_Imgs"
    detector = detector.Detector()
    for i in os.listdir(image_path):
        with Image.open(os.path.join(image_path,i)) as im:
            boxes=detector.detect(im)
            print("image size:",im.size)
            draw_rect(os.path.join(image_path,i),boxes)

            


# if __name__ == '__main__':
#     # 多张图片检测
#     image_path = r"test_images"
#     for i in os.listdir(image_path):
#         detector = Detector()
#         with Image.open(os.path.join(image_path,i)) as im: # 打开图片
#             # boxes = detector.detect(im)
#             print("----------------------------")
#             boxes = detector.detect(im)
#             print("size:",im.size)
#             imDraw = ImageDraw.Draw(im)
#             for box in boxes: # 多个框，没循环一次框一个人脸
#                 x1 = int(box[0])
#                 y1 = int(box[1])
#                 x2 = int(box[2])
#                 y2 = int(box[3])
#
#                 print((x1, y1, x2, y2))
#
#                 print("conf:",box[4]) # 置信度
#                 imDraw.rectangle((x1, y1, x2, y2), outline='red',width=2)
#                 #im.show() # 每循环一次框一个人脸
#             im.show()
#             # exit()