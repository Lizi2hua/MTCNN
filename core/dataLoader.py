from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import time

class FaceDataset(Dataset):
    def __init__(self,path):
        self.path=path
        self.dataset=[]
        self.dataset.extend(open(os.path.join(path,"positive.txt")).readlines())
        self.dataset.extend(open(os.path.join(path,"part.txt")).readlines())
        self.dataset.extend(open(os.path.join(path,"negative.txt")).readlines())
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        strs=self.dataset[item].strip().split(" ")
        #['positive/2.jpg', '1', '0.025889967637540454', '0.007281553398058253', '-0.061488673139158574', '0.27103559870550165']
        #置信度
        confidence=int(strs[1])
        # print(confidence)
        confidence=torch.Tensor([confidence]) #数据类型统一,torch.Tensor输入是序列，所以要加[]，不加数值是错的
        #偏移量
        offset=torch.Tensor([float(strs[2]),float(strs[3]),float(strs[4]),float(strs[5])])
        #img_data
        img_path=os.path.join(self.path,strs[0])
        to_tensor=transforms.ToTensor()
        img_data=to_tensor(Image.open(img_path))#归一化，NCHW

        return img_data,confidence,offset

# path=r"C:\Users\Administrator\Desktop\test500\12"
# dataset=[]
# start_time=time.time()
# dataset=FaceDataset(path)
# end_time=time.time()
# print(dataset[1])
# print(end_time-start_time)
