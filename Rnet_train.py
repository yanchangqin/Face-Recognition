from net import R_Net
from train import Trainer

if __name__ == '__main__':
    r_net = R_Net()
    trainer = Trainer(net=r_net,dataset_path=r'E:\MTCNN\celeba\24',save_path=r'E:\MTCNN\param1',save_param='para_rnet.pt')
    trainer.train()