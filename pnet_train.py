from net import P_Net
from train import Trainer

if __name__ == '__main__':
    p_net = P_Net()
    # trainer = Trainer(net=p_net,dataset_path=r'F:\MTCNN\test2\celeba\12',save_path='./para_pnet.pt')
    trainer = Trainer(net=p_net,dataset_path=r'E:\MTCNN\celeba\12',save_path=r'E:\MTCNN\param1',save_param='para_pnet.pt')
    trainer.train()