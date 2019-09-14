import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DATA
import torch.utils.data as data
from net import P_Net,R_Net,O_Net

class Trainer:
    def __init__(self,net,dataset_path,save_path,save_param,iscuda=True):
        self.net = net
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.save_param = save_param
        self.iscuda = iscuda

        if self.iscuda:
            self.net = self.net.cuda()
            print('GPU')

        self.loss_f1 = nn.BCELoss()
        self.loss_f2 = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())

        if os.path.exists(os.path.join(self.save_path,self.save_param)):
            # net.load(torch.load(self.save_path))
            self.net=torch.load(os.path.join(self.save_path,self.save_param))

    def train(self):
        face_dataset = DATA(self.dataset_path)
        dataloder = data.DataLoader(dataset=face_dataset,batch_size=500,shuffle=True,num_workers=4)
        for epoch in range(16,1000):
            for i,(img_data,conf,offset)in enumerate(dataloder):
                if self.iscuda:
                    img_data = img_data.cuda()
                    conf = conf.cuda()
                    offset = offset.cuda()
                img_data = img_data.permute(0,3,1,2)
                output_conf,output_offset = self.net(img_data)
                output_conf = output_conf.view(-1,1)
                # print(output_conf[0])
                # print(conf[0])
                output_offset = output_offset.view(-1,4)
                # print(output_conf.size())
                # print(output_offset.size())
                #选出置信度为0,1的标签做损失
                conf_index = torch.lt(conf,2)
                cond_mask = conf[conf_index]
                output_confmask = output_conf[conf_index]
                loss_conf = self.loss_f1(output_confmask,cond_mask)
                # print(output_confmask[0])
                # print(cond_mask[0])

                # 选出置信度为1,2的标签做损失
                offset_index = torch.gt(conf, 0)
                offset_mask= offset[torch.nonzero(offset_index)]
                output_offsetmask = output_offset[torch.nonzero(offset_index)]
                loss_offset = self.loss_f2(output_offsetmask,offset_mask)
                #总损失
                loss = loss_conf+loss_offset

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if i%50==0:
                    print(epoch)
                    print('总损失：',loss.item(),'置信度损失：',loss_conf.item(),'偏移量损失：',loss_offset.item())
            # torch.save(self.net,self.save_path)
            if not os.path.exists(os.path.join(self.save_path, str(epoch))):
                os.makedirs(os.path.join(self.save_path, str(epoch)))
            torch.save(self.net, os.path.join(self.save_path, str(epoch), self.save_param))




# p_net = P_Net()
# r_net = R_Net()
# o_net = O_Net()
# trainer = Trainer(net=o_net,dataset_path=r'F:\MTCNN\test2\celeba\48',save_path='./para_pnet.pt')
# trainer.train()


