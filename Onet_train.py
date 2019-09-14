from net import O_Net
from train import Trainer

if __name__ == '__main__':
    o_net = O_Net()
    trainer = Trainer(net=o_net,dataset_path=r'E:\MTCNN\celeba\48',save_path=r'E:\MTCNN\param1\17',save_param='para_onet.pt')
    trainer.train()