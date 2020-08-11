import numpy as np
from NNTools.IoU import iou
# 舍不得删的废案
# def NMS(boxes,thresh=0.2):
#        # The input data shoulde be like:[[Cofidence,top_left_x,top_left_y,botton_right_x,botton_right_y],[],[]]
#        boxes=np.array(boxes,dtype=float)
#        #进行降序排序
#        boxes=boxes[np.argsort(boxes[:,0])[::-1]]
#        # print(boxes)
#        #将boxes第一个元素与剩下元素计算Iou，大于阈值则删除，小于则保留
#        for i in range(len(boxes)):
#               j=i+1
#               # print("i=", i)
#               # print("boxed len",len(boxes))
#               if i >=len(boxes):
#                      break
#               compare_base_box=boxes[i][1:]
#               # print(max_confidence_box)
#               for item in boxes[j:]:
#                      # print(j)
#                      # print(item)
#                      item=item[1:]
#                      _,_,IOU=iou(item,compare_base_box)
#                      # print(IOU)
#                      #因为np.delele删除后的boxes维度会缩小一行， 所以删除时，遍历指针j相当于+1
#                      if IOU>=thresh:
#                             boxes=np.delete(boxes,j,0)
#                             # print('delte!')
#                      else:j+=1
#                      # print("________")
#               # print(boxes)
#        return boxes

def NMS(boxes,thresh=0.2,is_Min=False):
       """
       :param boxes: 输入
       :param thresh: 超参，阈值，用于判断是否舍弃数据
       :param is_Min: IoU函数的参数
       :return: NMS后的数据
       """
       #如果网络没有输出建议框,box.shape=(0,)
       if boxes.shape[0]==0:
              return np.array([])
       buffer=[]
       boxes_sorted=boxes[np.argsort(boxes[:,0])[::-1]]#降序
       # print(boxes_sorted.shape)
       # print(boxes_sorted)
       # while boxes_sorted.shape[0]>1:
       while boxes_sorted.shape[0] >=1:
           #取第一个框
              a_box=boxes_sorted[0]

           #取剩下的框
              b_boxes=boxes_sorted[1:]
           #第一个框必定是保留的
              buffer.append(a_box)
           #比较IoU，大于阈值的框去掉，用bool索引
              IOU=iou(a_box,b_boxes,is_Min)
              mask=np.where(IOU<thresh)

           #用bool索引得到得array来替代上一次迭代保留的框的array数据
              boxes_sorted=b_boxes[mask]


       # if boxes_sorted.shape[0]>0:
       #        buffer.append(boxes_sorted[0])#boxes_sorted=[[]]的形状

       return np.stack(buffer)
       # return buffer








# boxes=np.array([])
# print(boxes.shape)
# b=NMS(boxes)
# print(b)


# boxes=np.array([[0.97,133,80,225,265],
#        [0.89,157,69,261,238],
#        [0.85,148,132,238,282],
#        [0.70,105,112,195,303],
#        [0.69,88,50,187,193],
#        [0.70,316,209,378,312],
#        [0.50,298,173,348,340],
#        [0.90,490,67,563,251],
#        [0.70,446,46,526,181],
#        [0.79,533,41,622,175],
#        [0.85,429,87,619,216]])
# box=NMS(boxes,thresh=0.1)
# print(box)

