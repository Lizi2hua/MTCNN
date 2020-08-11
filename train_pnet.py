from train import Trainer
from net import PNet
if __name__ == '__main__':
    save_path="saved_models/pnet.pt"
    dataset_path=r"C:\gened_data\12"
    sum_path='./pnet_logs'
    net=PNet()
    train=Trainer(net=net,save_path=save_path,dataset_path=dataset_path,sum_path=sum_path)
    train()
