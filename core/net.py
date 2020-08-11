import torch
import torch.nn as nn
import torch.nn.functional as F

class PNet(nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.feature_layer=nn.Sequential(
            #12*12*3
            nn.Conv2d(in_channels=3,out_channels=10,kernel_size=3,padding=1,stride=1),# conv3*3,s=1
            nn.PReLU(),
            #12*12*10
            nn.MaxPool2d(kernel_size=3,stride=2),#kernel_size=3,stride=2
            #5*5*10
            nn.Conv2d(in_channels=10,out_channels=16,kernel_size=3,stride=1),#conv3*3,s=1
            nn.PReLU(),
            #3*3*16
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1),#conv3*3,s=1
            nn.PReLU(),
            #1*1*32
            )
        self.dect_layer=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride=1)
        )
        self.offset_layer=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=4,kernel_size=1,stride=1)
        )

    def forward(self,x):#x:[N,C,H,W]
        out=self.feature_layer(x)
        dect_out=self.dect_layer(out) #output:confidence
        dect_out=torch.sigmoid(dect_out) #使用BCE_loss之前要进行sigmoid激活
        offset_out=self.offset_layer(out) #output:offset
        return dect_out,offset_out

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.feature_layer=nn.Sequential(
            #24*24*3
            nn.Conv2d(in_channels=3,out_channels=28,kernel_size=3,stride=1,padding=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),#kernel_size=2,s=2
            #11*11*28
            nn.Conv2d(in_channels=28,out_channels=48,kernel_size=3,stride=1),#conv3*3,s=1
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),#kenel_size=2,s=1

            #4*4*48
            nn.Conv2d(in_channels=48,out_channels=64,kernel_size=2,stride=1),#conv2*2,s=1
            nn.PReLU(),
        )
        self.fc_layer=nn.Sequential(
             #3*3*64
            nn.Linear(in_features=3*3*64,out_features=128),
            nn.PReLU()
        )
        self.dect_layer=nn.Sequential(
            #128
            nn.Linear(in_features=128,out_features=1),#[N,1]
        )
        self.offset_layer=nn.Sequential(
            #128
            nn.Linear(in_features=128,out_features=4),#[N,4]
        )

    def forward(self,x):#output:[N,V]
        x=self.feature_layer(x)
        x=x.reshape(-1,3*3*64) #[N,V],V=3*3*64
        out=self.fc_layer(x)
        dect_out=self.dect_layer(out)
        dect_out=torch.sigmoid(dect_out) #BCEloss前需要sigmoid激活
        offset_out=self.offset_layer(out)
        return dect_out,offset_out


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.feature_layer=nn.Sequential(
            #48*48*3
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1),#conv 3*3,s=1
            nn.PReLU(),
            #48*48*32
            nn.MaxPool2d(kernel_size=3,stride=2),
            #23*23*32
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1),
            nn.PReLU(),
             #21*21*64
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.PReLU(),
            #10*10*64
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
            nn.PReLU(),
            #8*8*64
            nn.MaxPool2d(kernel_size=2,stride=2),
            #4*4*64
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=2,stride=1),
            nn.PReLU()
            #3*3*128,[N,128,3,3]
        )
        self.fc_layer=nn.Sequential(
            nn.Linear(3*3*128,256)#[N,256]
        )
        self.dect_layer=nn.Sequential(
            nn.Linear(256,1)
        )
        self.offset_layer=nn.Sequential(
            nn.Linear(256,4)
        )

    def forward(self,x):
        x=self.feature_layer(x) #[N,128,3,3]
        x=x.reshape(-1,128*3*3)#[N,128*3*3]
        out=self.fc_layer(x)
        dect_out=self.dect_layer(out)
        dect_out=torch.sigmoid(dect_out)
        offset_out=self.offset_layer(out)
        return dect_out,offset_out


#test
# a=torch.randn(64,3,12,12)
# b=torch.randn(64,3,24,24)
# c=torch.randn(64,3,48,48)
#
# net1=PNet()
# net2=RNet()
# net3=ONet()
# a1,a2=net1(a)
# print(a1.shape,"__",a2.shape )


