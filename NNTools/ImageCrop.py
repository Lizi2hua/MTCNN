import numpy as np
import os
import json
import traceback
from PIL import Image
from tqdm import tqdm
from  draw_rectangle import *
from IoU import  *
import time
import random

#配置信息
# anno_file=r"C:\Users\Administrator\Desktop\gened_data\label"
img_file=r"C:\Users\Administrator\Desktop\test500\img500"
save_path=r"C:\Users\Administrator\Desktop\test500"
label_txt=r'C:\Users\Administrator\Desktop\test500\500_label.txt'
# anno_file=r"C:\Users\李梓桦\Desktop\test_data\label"
# img_file=r"C:\Users\李梓桦\Desktop\test_data\img"
# save_path=r"C:\Users\李梓桦\Desktop\test_data"
# label_txt=r'C:\Users\李梓桦\Desktop\test_data\label.txt'

face_size=[12,24,48]
# face_size=[48]

# 检查文件是否存在
for size in face_size:
    if os.path.exists(os.path.join(save_path,str(size))):
        raise FileExistsError('File already exist,please delete!')
#生成图片
for size in face_size:
    # 裁剪的是正方形人脸，这样缩放时人脸的变形小
    print("generating {}*{} size imgae".format(size,size))
    #创建正例，负例，part人脸文件夹
    pos_save_dir=os.path.join(save_path,str(size),"positive")
    part_save_dir=os.path.join(save_path,str(size),"part")
    neg_save_dir=os.path.join(save_path,str(size),"negative")

    for dir_path in [pos_save_dir,part_save_dir,neg_save_dir]:
        os.makedirs(dir_path)

    # 样本标本用‘.txt’文件存
    pos_anno_file=os.path.join(save_path,str(size),"positive.txt")
    part_anno_file=os.path.join(save_path,str(size),"part.txt")
    neg_anno_file=os.path.join(save_path,str(size),"negative.txt")
    # 计数器，给标签和文件命名
    positive_count=0
    negative_count=0
    part_count=0
    # 以“write"的方式打开文件
    try:
        pos_anno_file=open(pos_anno_file,"w")
        part_anno_file=open(part_anno_file,"w")
        neg_anno_file=open(neg_anno_file,"w")

        for line in tqdm(open(label_txt),desc="希望图片没事",ncols=10):

            try:
                strs=line.split()
                img_path=strs[0]
                img_path=os.path.join(img_file,img_path)

                with Image.open(img_path) as img:
                    img_w,img_h=img.size
                    #这行代码是[左上角x,左上角y,右下角x,右下角y]
                    # topx=int(strs[1])
                    # topy=int(strs[2])
                    # bottonx=int(strs[3])
                    # bottony=int(strs[4])
                    # w=bottonx-topx
                    # h=bottony-topy
                    # rect=[1.0,topx,topy,bottonx,bottony]
                    # draw_rect(img_path,[rect])

                    #这行代码是celeba去除前两行得标注信息
                    topx = float(strs[1].strip()) # 取2nd个值去除两边的空格，再转车float型
                    topy = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())
                    bottonx = float(topx + w)
                    bottony = float(topy + h)



                    if max(w,h)<40 \
                            or topx<0 or topy<0 or w<0 or h<0:
                    #图片最大48*48，所以应该切到40或以上,不满足的舍弃，下面的保证数据是准确的
                            continue
                    boxes=[[topx,topy,bottonx,bottony]]
                    #计算人脸中心坐标
                    cx=topx+w/2#cx:center x
                    cy=topy+h/2#cy:center y
                    # 生成部分人脸和人脸的图片

                    for _ in range(3):
                         #直接使用np.random.randint似乎有bug

                        w_=np.random.randint(-w*0.25,w*0.26)#横向随机偏移0.2w
                        h_=np.random.randint(-h*0.25,h*0.26) #纵向随机偏移0.2h

                        cx_=cx+w_
                        cy_=cy+h_
                        #生成与偏移后的中心点的偏移值，使用该偏移值作为框的宽高（w=h)
                        offset_len=np.random.randint(int(min(w,h)*0.8),np.ceil(max(w,h)*1.25))
                        x1_=np.max(cx_-offset_len/2,0)#防止框的左上角出现在图片左上角之外
                        y1_=np.max(cy_-offset_len/2,0)
                        x2_=x1_+offset_len
                        y2_=y1_+offset_len

                        crop_box=[x1_,y1_,x2_,y2_]
                        # rect=[1.0,x1_,y1_,x2_,y2_]
                        # draw_rect(img_path,[rect])
                        #计算偏移值
                        offset_x1=(topx-x1_)/offset_len
                        offset_y1=(topy-y1_)/offset_len
                        offset_x2=(bottonx-x2_)/offset_len
                        offset_y2=(bottony-y2_)/offset_len

                       # rect=[1.0,x1_,y1_,x2_,y2_]
                        # draw_rect(img_path,[rect])
                        #从原图裁切
                        crop_face=img.crop(crop_box)
                        #缩放到12*12，24*24，48*48
                        resized_face=crop_face.resize((size,size),Image.ANTIALIAS)
                    #
                        # print("crop_boxes:",crop_box)
                        # print(resized_faces)
                        # print(offset_X2)
                        # exit()
                        IOU=iou(crop_box,np.array(boxes))
                        if IOU>0.6:
                            pos_anno_file.write("positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    positive_count,1,offset_x1,offset_y1,offset_x2,offset_y2
                                ))
                            pos_anno_file.flush()
                            resized_face.save(os.path.join(pos_save_dir,"{0}.jpg".format(positive_count)))
                            positive_count+=1
                        if IOU > 0.4:
                            part_anno_file.write("part/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                part_count, 2, offset_x1, offset_y1, offset_x2, offset_y2
                            ))
                            part_anno_file.flush()
                            resized_face.save(os.path.join(part_save_dir, "{0}.jpg".format(part_count)))

                            part_count += 1
                            #负样本单独生成，这里只是占位
                            if IOU<0.29:
                                pass
                        #time.sleep(0.002)
                    #负样本生成
                    for i in range(3):#数量一般和上面一样

                        offset_len=np.random.randint(size,min(img_w,img_h)/2)# 偏移量
                        x1_=np.random.randint(0,img_w-offset_len)#生成左上角的坐标,生成的patch都在左上角
                        y1_=np.random.randint(0,img_h-offset_len)
                        x1_=max(0,x1_)
                        y1_=max(0,y1_)
                        crop_box1=np.array([x1_,y1_,x1_+offset_len,y1_+offset_len])

                        offset_top=np.random.randint(0.75*size,1.25*size)
                        x2_=np.random.randint(bottonx,bottonx+offset_top)#生成左上角的坐标，使得左上角坐标落在图片下部分
                        y2_=np.random.randint(bottony,bottony+offset_top)

                        xx2_=min(img_w-1,bottonx+offset_len)
                        yy2_=min(img_h-1,bottony+offset_len)
                        crop_box2=[x2_,y2_,xx2_,yy2_]


                        crop_boxes=[crop_box1,crop_box2]
                        idx=np.random.randint(0,2)
                        crop_box=crop_boxes[idx]

                        # rect=[1.0,crop_box[0],crop_box[1],crop_box[2],crop_box[3]]
                        # draw_rect(img_path,[rect])



                        if iou(crop_box,np.array(boxes))<0.29:
                            face_crop = img.crop(crop_box)  # 抠图
                            face_resize = face_crop.resize((size, size), Image.ANTIALIAS)  # ANTIALIAS：平滑,抗锯齿

                            neg_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0 0 \n".format(negative_count, 0))
                            neg_anno_file.flush()
                            face_resize.save(os.path.join(neg_save_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1


            except Exception as e:
                traceback.print_exc()#获取异常信息
    finally:
        pos_anno_file.close()
        neg_anno_file.close()
        part_anno_file.close()
