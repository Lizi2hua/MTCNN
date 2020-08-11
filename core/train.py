import os,torch,time
from torch import  nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as opt
from dataLoader import FaceDataset
from tqdm import tqdm
BATCH_SIZE=512
EPOCH=100000

class Trainer:
    def __init__(self,net,save_path,dataset_path,sum_path,is_Cuda=True,):
        self.net=net
        self.save_path=save_path
        self.dataset_path=dataset_path
        self.isCuda=is_Cuda
        self.sum_path=sum_path

        if self.isCuda:
            self.net.cuda()#将网络放在GPU上

        #创建损失函数
        #1.二分类损失
        self.cls_loss_fn=nn.BCELoss()
        #2.回归损失（gt和out的L2差距）
        self.reg_loss_fn=nn.MSELoss()

        self.opt=opt.Adam(self.net.parameters())
        self.summary=SummaryWriter(self.sum_path)

        # 加载网络训练结果，模型持久化
        if os.path.exists(self.save_path):
            net.load_state_dict(torch.load(self.save_path))



    def __call__(self):
        print("So,let give the data a hard time!")
        faceDataset=FaceDataset(self.dataset_path)
        dataloader=DataLoader(faceDataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4,
                                  drop_last=True)
        for epoch in tqdm(range(EPOCH),desc="希望数据没事",ncols=70):
            start_time=time.time()
        # while True:

            sum_loss=0.
            cls_loss=0.
            reg_loss=0.
            for i,(img_data,coffidence,offset) in enumerate(dataloader):
                # print("training")
                start_time=time.time()
                if self.isCuda:
                    img_data=img_data.cuda() #[BATCH_SIZE,3,W,W]
                    coffidence=coffidence.cuda() #[BATCH_SIZE,1，1，1],[BATCH_SIZE,1]
                    offset=offset.cuda()#[BATCH_SIZE,4，1，1],[BATCH_SIZE,4]

                    #网络输出
                _output_coffidence,_output_offset=self.net(img_data)
                #R-Net输出的的信息在通道上，需要转换到形状维度（H,W的维度）,其它的无变化
                output_coffidence=_output_coffidence.reshape(-1,1) #R-Net:[BATCH_SIZE,1]
                output_offset=_output_offset.reshape(-1,4) #R-Net：[BATCH_SIZE,4]
                # print("have output")
                #计算损失
                #1.计算分类损失（置信度）
                coffidence_idx=torch.where(coffidence<2)[0]
                coffidence_loss=self.cls_loss_fn(output_coffidence[coffidence_idx],
                                                     coffidence[coffidence_idx]) #样本中排除置信度大于2的（part样本），不参与分类loss计算
                #2.计算回归损失
                offset_idx=torch.where(coffidence>0)[0]
                regression_loss=self.reg_loss_fn(output_offset[offset_idx],
                                                 offset[offset_idx]) #样本中置信度维0的没有偏移量，不参与回归loss计算
                # print("计算损失")
                #3.损失求和
                loss=coffidence_loss+regression_loss

                #三件套
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                # print("损失反传结束")

                sum_loss+=loss.cpu().data.numpy()
                cls_loss+=coffidence_loss.cpu().data.numpy()
                reg_loss+=regression_loss.cpu().data.numpy()
                end_time=time.time()
                if i%200==0:
                    print("loss{}:{},cost {}s".format(i,loss,(end_time-start_time)))
                # print("一个数据结束")
            end_time=time.time()
            print("{}epoch 结束,耗时{}".format(epoch,(end_time-start_time)))
            avg_loss=sum_loss/len(dataloader)
            avg_cls_loss=cls_loss/len(dataloader)
            avg_reg_loss=reg_loss/len(dataloader)

            # 记录
            self.summary.add_scalar('jiont_loss',avg_loss,epoch)
            self.summary.add_scalars('other_losses',{"classi_loss":avg_cls_loss,"reges_loss":avg_reg_loss},epoch)
            print("\033[1;31mjoint loss:{},class_loss:{},reg_Losss{}\033[0m".format(avg_loss,cls_loss,reg_loss))
            # 存模型
            # epoch+=1
            # if epoch%1==0:

            print('save the {} eopch model.....'.format(epoch))
            torch.save(self.net.state_dict(),self.save_path)
            print('save successfully!')











# a=0
# # for i in tqdm(range(EPOCH),desc="nn!",ncols=70):
# #     a+=i
# #     time.sleep(0.001)
# # print(a)
# a=torch.Tensor([[1],[2],[1],[0]])
# mask=torch.where(a<2)
# print(mask)



