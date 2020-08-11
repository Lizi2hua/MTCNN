import numpy as np
# IoU in Rectangle Condition
def iou(box1,boxes,is_Min=False):
    """
    :param box1: 输入的框，形式为 [top_left_x,top_left_y,botton_right_x,botton_right_y]
    :param boxes: 输入的框，形式为 [[top_left_x,top_left_y,botton_right_x,botton_right_y],[]]
    :param is_Min: 是否用最小面积计算IoU
    :return: 交并比
    """
    #计算交集面积
    inter_top_left_x=np.maximum(box1[0],boxes[:,0])
    inter_top_left_y=np.maximum(box1[1],boxes[:,1])
    inter_botton_right_x=np.minimum(box1[2],boxes[:,2])
    inter_botton_right_y=np.minimum(box1[3],boxes[:,3])
    # 判断是否满足形成交集的条件:右下角的x值大于左上角的x值，右下角的y值大于左上角的y值（即右下角的坐标应该在右下角，、
    # 左上角的坐标应该在左上角）
    # w=右下角的x值-左上角的x值，h=右下角的y值-左上角的y值
    w=np.maximum(0,inter_botton_right_x-inter_top_left_x)
    h=np.maximum(0,inter_botton_right_y-inter_top_left_y)
    inter_area=w*h
    #计算并集面积 union_area=box1_area+box2_area-inter_area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    if is_Min:
        IOU=np.true_divide(inter_area,np.minimum(box1_area,boxes_area))
    else:
        IOU=np.true_divide(inter_area,(box1_area+boxes_area-inter_area))
    return IOU



