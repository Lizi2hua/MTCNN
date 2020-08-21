import os, time, torch
import numpy as np
from PIL import Image
from PIL import ImageDraw
from torchvision.transforms import transforms
from NNTools.draw_rectangle import draw_rect
from NNTools.NMS import NMS
import matplotlib.pyplot as plt
from core import net

# 网络超参数
# super params: cf，置信度，iou，交并比阈值
p_cf = 0.5
p_iou = 0.2

r_cf = 0.6
r_iou = 0.3

o_cf = 0.9999
o_iou = 0.9

# 配置
p_pt = r'saved_models\pnet.pt'
r_pt = r'saved_models\rnet.pt'
o_pt = r'saved_models\onet.pt'


class Detector:
    def __init__(self, isCuda=True):
        self.isCuda = isCuda
        self.pnet = net.PNet()
        self.rnet = net.RNet()
        self.onet = net.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()
        print("load nets.....")
        self.pnet.load_state_dict(torch.load(p_pt))
        self.rnet.load_state_dict(torch.load(r_pt))
        self.onet.load_state_dict(torch.load(o_pt))
        self.__image_trans = transforms.ToTensor()
        print("nets loaded successfully!")

    # pnet检测函数
    def __p_detect(self, image):
        boxes = []
        img = image
        w, h = img.size
        # 以最小边长为图像金字塔结束的判定条件
        min_side = min(w, h)
        scale = 1
        # 每缩放一次就放进pnet做一次检测
        while min_side > 12:
            # 将图像转为tensor，并做归一化处理
            img_data = self.__image_trans(img)
            if self.isCuda:
                img_data = img_data.cuda()
            # 对图片tensor进行升维，变成[N,C,H,W],CNN网络的输入必须是[N,C,H,W]
            # img_data.unsqueeze_(0)#凡是以‘_’为后缀的都是能修改原对象的
            img_data = torch.unsqueeze(img_data, 0)

            _cf, _offset = self.pnet(img_data)  # 数据为[N,C,H,W]，H,W的值于输入的图片大小有关
            cf = _cf[0][0].cpu().detach()  # 分组卷积，置信度,输出：[N,1,h,w]。变为了[h,w]
            offset = _offset[0].cpu().detach()  # 分组卷积，偏移量，输出：[N,4,h,w],变为了[4,H,W]
            idxs = torch.nonzero(torch.gt(cf, p_cf), as_tuple=False)  # 网络输出的置信度与阈值比较，得到bool列表，最后得到索引
            # 由于pnet的输出的框太多，不适合用for循环



            for idx in idxs:
                boxes.append(self.__box(idx, offset, cf[idx[0], idx[1]], scale))  # 坐标反算回原图

            # 缩放图片
            scale *= 0.7
            _w = int(w * scale)
            _h = int(h * scale)
            img = img.resize((_w, _h))
            min_side = min(_w, _h)

        return NMS(np.array(boxes), p_iou)  # [cf,x1,y1,x2,y2]

    # rnet检测函数
    def __r_detect(self, image, pnet_boxes):
        _img_crop = []
        _pnet_boxes = self.convert_to_square(pnet_boxes)
        # 根据框从原图抠图
        for _box in _pnet_boxes:
            _x1 = int(_box[1])
            _y1 = int(_box[2])
            _x2 = int(_box[3])
            _y2 = int(_box[4])
            # print(_x1,_y1,_x2,_y2)
            # print(_box[0])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((24, 24))
            img_data = self.__image_trans(img)
            _img_crop.append(img_data)
        img_crop = torch.stack(_img_crop)  # [N,V],N候选框的个数

        if self.isCuda:
            img_crop = img_crop.cuda()

        _cf, _offset = self.rnet(img_crop)
        # _cf:[N,1],_offset:[N,4]
        cf = _cf.cpu().detach()
        offset = _offset.cpu().detach()
        # print('rnet offset:',offset)
        # print('cf:',cf)

        boxes = []
        idxs, _ = np.where(cf > r_cf)  # 比较网络输出置信度和设置得置信度，得到索引:[n,v]
        # print('idxs:',idxs)
        for idx in idxs:
            # 获取到变为正方形框得
            _box = _pnet_boxes[idx]
            _x1 = int(_box[1])
            _y1 = int(_box[2])
            _x2 = int(_box[3])
            _y2 = int(_box[4])
            # print(_x1,_y1,_x2,_y2)
            # print(_x2-_x1,_y2-_y1)
            # exit()

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            # print('rnet x1：{},y1：{},x2：{},y2：{}'.format(x1,y1.x2,y2))
            boxes.append([cf[idx][0], x1, y1, x2, y2])
        # print('rnet boxes:',boxes)
        return NMS(np.array(boxes), r_iou)

    # onet检测函数和rnet差别不大
    def __o_detect(self, image, rnet_boxes):
        #[cf,x1,y1,x2,y2]
        isMin = True
        _img_crop = []
        _rnet_boxes = self.convert_to_square(rnet_boxes)
        # 根据框从原图抠图
        for _box in _rnet_boxes:
            _x1 = int(_box[1])
            _y1 = int(_box[2])
            _x2 = int(_box[3])
            _y2 = int(_box[4])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_trans(img)
            _img_crop.append(img_data)
        img_crop = torch.stack(_img_crop)
        if self.isCuda:
            img_crop = img_crop.cuda()  # [N,V],N候选框得个数

        _cf, _offset = self.onet(img_crop)
        # _cf:[N,1],_offset:[N,4]
        cf = _cf.cpu().detach()
        offset = _offset.cpu().detach()

        boxes = []
        idxs, _ = np.where(cf > r_cf)  # 比较网络输出置信度和设置得置信度，得到索引:[n,v]

        for idx in idxs:
            # 获取到变为正方形框得
            _box = _rnet_boxes[idx]  # 以r网络输出的结果为候选框
            _x1 = int(_box[1])
            _y1 = int(_box[2])
            _x2 = int(_box[3])
            _y2 = int(_box[4])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]

            boxes.append([cf[idx][0], x1, y1, x2, y2])
        return NMS(np.array(boxes), r_iou,is_Min=isMin)

    # 特征反算
    def __box(self, start_index, offset, cf, scale, stride=2, side_len=12):
        """
        特征反算过程和正向计算特征图的原理差不多，一个正向计算，一个反向计算
        :param start_index: 候选区域的的左上角坐标（在pnet的输出就是一个坐标值，候选区域是个点）
        【注意像素点的坐标表示为(x,y),而tensor.numpy,list的索引为（y,x）】
        :param offset: pnet输出的偏移量
        :param cf:
        :param scale:缩放比例
        :param stride:步长为2，是网络中所有步长的乘积
        :param side_len:pnet等效一个12*12的卷积核
        :return:原图上的坐标
        topx=indexx*stride
        topy=indey*stride
        bottonx=indexx*stride+ksizex-1
        bottony=indexy*stride+ksizey-1
        """
        # 反算到原图上的候选区域坐标
        _x1 = (start_index[1].float() * stride) / scale
        _y1 = (start_index[0].float() * stride) / scale
        _x2 = (start_index[1].float() * stride + side_len) / scale - 1
        _y2 = (start_index[0].float() * stride + side_len) / scale - 1

        oh = ow = side_len / scale  # 这样算更精确
        # 计算映射回原图的候选区域
        # 得到实际框相对于候选区域的偏移量
        _offset = offset[:, start_index[0], start_index[1]]  # [4,H,W]
        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [cf, x1, y1, x2, y2]  # NMS输入的形状如此

    # 由于R,O网络输入必须是矩形，倘若简单的使用resize，会破坏原图的特征。可以将pnet输出的框扩张为正方形
    def convert_to_square(self, bbox):
        '''
        将原始框变为正方形框
        :param bbox: 原始框
        :return: 正方形框
        '''
        #[cf,x1,y1,x2,y2]
        square_bbox = bbox.copy()
        if bbox.shape[0] == 0:
            return np.array([])
        # bbox:[N,V],N指N个框
        w = bbox[:, 3] - bbox[:, 1]
        h = bbox[:, 4] - bbox[:, 2]
        # 以最长边为基准扩张
        max_side = np.maximum(h, w)
        # 先计算左上角坐标，让后左上角加最大边的右下角
        square_bbox[:, 1] = (bbox[:, 1] + w * 0.5) - max_side * 0.5  #()内计算中心
        square_bbox[:, 2] = (bbox[:, 2] + h * 0.5) - max_side * 0.5
        square_bbox[:, 3] = bbox[:, 1] + max_side
        square_bbox[:, 4] = bbox[:, 2] + max_side

        return square_bbox

    def detect(self, image):
        # pnet检测
        start_time = time.time()
        pnet_boxes = self.__p_detect(image)
        end_time = time.time()
        pnet_time = end_time - start_time
        print("pnet cost\033[1;31m{}s\033[0m".format(pnet_time))
        if pnet_boxes.shape[0] == 0:  # [m,n],NMS函数决定
            return np.array([])  # 如果pnet没有检测到人脸，结束并返回一个None值
        # return pnet_boxes

        # rnet检测
        start_time = time.time()
        rnet_boxes = self.__r_detect(image, pnet_boxes)  # 根据pnet的输出从原图抠图，得到类似于[N,C,H,W]类型。N是boxes_nums
        end_time = time.time()
        rnet_time = end_time - start_time
        print("rnet cost\033[1;31m{}s\033[0m".format(rnet_time))
        if rnet_boxes.shape[0] == 0:
            print("rnet cant find anyone!")
            return np.array([])
        # return rnet_boxes

        # onet检测
        start_time = time.time()
        onet_boxes = self.__o_detect(image, rnet_boxes)
        end_time = time.time()
        onet_time = end_time - start_time
        print("onet cost\033[1;31m{}s\033[0m".format(onet_time))
        # 三网络总时间
        t_sum = pnet_time + rnet_time + onet_time

        print("total cost\033[1;31m{}s\033[0m".format(t_sum))
        if onet_boxes.shape[0] == 0:
            print("onet cant find anyone!")
            return np.array([])

        return onet_boxes
