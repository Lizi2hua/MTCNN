import os
import torch
from NNTools.NMS import NMS
import numpy as np

# a=torch.load('saved_models\pnet.pt')
# print(a)
# time1=3.
# time2=2.0
# time3=4
# print(' a \033[1;37;40mtotal cost:{}s\033[0m'.format(time3))
# print("total cost\033[1;31m{}s\033[0m,pnet:\033[1;31m{}s\033[0m,onet:\033[1;31m{}s\033[0m".format(time3,time3,time3,time3))
a=torch.Tensor([[0.6,0.5],[0.9,0.8],[0.7,0.6]])
a2=torch.Tensor([[[0.2,0.2],[0.3,0.3],[0.4,0.4]],[[0.5,0.5],[0.6,0.6],[0.7,0.7]],[[0.8,0.8],[0.9,0.9],[0.0,0.0]]])
a=torch.tensor([[0.5],[0.4],[0.3],[0.6],[0.7],[0.8]])
b,_=torch.where(a>0.6)

# c=a2[b]
# z=torch.nonzero(b,as_tuple=False)
print(a[b])
# print(z)
# print(z[0])
a=boxes=np.array([[0.97,133,80,225,265],
       [0.89,157,69,261,238],
       [0.85,148,132,238,282],
       [0.70,105,112,195,303],
       [0.69,88,50,187,193],
       [0.70,316,209,378,312],
       [0.50,298,173,348,340],
       [0.90,490,67,563,251],
       [0.70,446,46,526,181],
       [0.79,533,41,622,175],
       [0.85,429,87,619,216]])
box=NMS(boxes,thresh=0.1)
print(box)

