from train import Trainer
from net import ONet
if __name__ == '__main__':
    save_path="saved_models/onet.pt"
    dataset_path=r"C:\gened_data\48"
    sum_path='./onet_logs'
    net=ONet()
    train=Trainer(net=net,save_path=save_path,dataset_path=dataset_path,sum_path=sum_path)
    train()
