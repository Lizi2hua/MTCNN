from core.train import Trainer
from core.net import RNet
if __name__ == '__main__':
    save_path="saved_models/rnet.pt"
    dataset_path=r"C:\gened_data\24"
    sum_path='./rnet_logs'
    net=RNet()
    train=Trainer(net=net,save_path=save_path,dataset_path=dataset_path,sum_path=sum_path)
    train()
