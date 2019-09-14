import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
# import iou_nms
import nms_iou
import matplotlib.pyplot as plt
import net
from torchvision import transforms
import time
import PIL.ImageFont as font
import os
import cv2


class Detector:

    def __init__(self, pnet_param=r'E:\MTCNN\param1\10\para_pnet.pt',rnet_param=r'E:\MTCNN\param1\15\para_rnet.pt',onet_param=r'E:\MTCNN\param1\15\para_onet.pt',isCuda=True):
        self.isCuda = isCuda

        # self.pnet = net.P_Net()
        # self.rnet = net.R_Net()
        # self.onet = net.O_Net()



        # self.pnet.load_state_dict(torch.load(pnet_param))
        # self.rnet.load_state_dict(torch.load(rnet_param))
        # self.onet.load_state_dict(torch.load(onet_param))
        self.p_net = torch.load(pnet_param)
        self.r_net = torch.load(rnet_param)
        self.o_net = torch.load(onet_param)

        # if self.isCuda:
        #     self.p_net.cuda()
        #     self.r_net.cuda()
        #     self.o_net.cuda()

        self.p_net.eval()
        self.r_net.eval()
        self.o_net.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def detect(self, image):

        start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)
        # print("111")
        if pnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_pnet = end_time - start_time
        # return pnet_boxes

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)
        # print( rnet_boxes)
        # print('222')
        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time
        # return rnet_boxes

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)
        # print('333')
        if onet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = nms_iou.convert_to_square(pnet_boxes)
        # _x1 = _pnet_boxes[:,0]
        # _y1 = _pnet_boxes[:,1]
        # _x2 = _pnet_boxes[:,2]
        # _y2 = _pnet_boxes[:,3]
        # w_ = _x2-_x1
        # h_ = _y2-_y1
        # print(w_,h_)
        # exit()
        # image_array = np.array(image)
        # print(image_array.shape)
        # imgg = image_array[_y1:_x1,:,:]
        # print(imgg)
        # img = Image.fromarray(imgg)
        # img.show()
        # # img = image.crop((_x1, _y1, _x2, _y2))
        # # print(img)
        # exit()
        # img = img.resize((24, 24))
        # img_data = self.__image_transform(img) - 0.5
        # _img_dataset.append(img_data)
        ''''''
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])


            img = image.crop((_x1, _y1, _x2, _y2))
            # print(img)
            # exit()
            img = img.resize((24, 24))
            # plt.imshow(img)
            # plt.pause(1)
            img_data = self.__image_transform(img)-0.5
            _img_dataset.append(img_data)
            # print('_img_dataset',_img_dataset)

        img_dataset =torch.stack(_img_dataset)
        # print('img_dataset',img_dataset.shape)
        if self.isCuda:
            img_dataset = img_dataset.cuda()
        # with torch.no_grad():
        _cls, _offset = self.r_net(img_dataset)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        # print("R网络置信度：",cls)

        boxes = []
        idxs, _ = np.where(cls > 0.6)
        boxes_ = _pnet_boxes[idxs]
        _x1 = boxes_[:,0]
        _y1 = boxes_[:,1]
        _x2 = boxes_[:,2]
        _y2 = boxes_[:,3]
        ow = _x2 - _x1
        oh = _y2 - _y1
        x1 = _x1 + ow * offset[idxs][:,0]
        y1 = _y1 + oh * offset[idxs][:,1]
        x2 = _x2 + ow * offset[idxs][:,2]
        y2 = _y2 + oh * offset[idxs][:,3]
        boxes = np.stack((x1, y1, x2, y2, cls[idxs][:, 0]),axis=1)
        # print(boxes)
        # exit()
        '''.......'''
        # for idx in idxs:
        #     _box = _pnet_boxes[idx]
        #     _x1 = int(_box[0])
        #     _y1 = int(_box[1])
        #     _x2 = int(_box[2])
        #     _y2 = int(_box[3])
        #
        #     ow = _x2 - _x1
        #     oh = _y2 - _y1
        #
        #     x1 = _x1 + ow * offset[idx][0]
        #     y1 = _y1 + oh * offset[idx][1]
        #     x2 = _x2 + ow * offset[idx][2]
        #     y2 = _y2 + oh * offset[idx][3]
        #     # imgs = image.crop((x1,y1,x2,y2))
        #     # plt.imshow(imgs)
        #     # plt.pause(1)
        #     self.cls = cls[idx][0]
        #     boxes.append([x1, y1, x2, y2, cls[idx][0]])

        boxb = nms_iou.nms(np.array(boxes), 0.4)

        return boxb

    def __onet_detect(self, image, rnet_boxes):
        _img_dataset = []
        _rnet_boxes = nms_iou.convert_to_square(rnet_boxes)

        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])
            # imDraw = ImageDraw.Draw(image)
            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            # plt.imshow(img)
            # plt.pause(1)
            img_data = self.__image_transform(img)-0.5
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()
        with torch.no_grad():
            _cls, _offset = self.o_net(img_dataset)
            # print("O网络置信度：",_cls)

        cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()
        boxes = []
        idxs, _ = np.where(cls > 0.9)

        boxes_ = _rnet_boxes[idxs]
        _x1 = boxes_[:, 0]
        _y1 = boxes_[:, 1]
        _x2 = boxes_[:, 2]
        _y2 = boxes_[:, 3]
        ow = _x2 - _x1
        oh = _y2 - _y1
        x1 = _x1 + ow * offset[idxs][:, 0]
        y1 = _y1 + oh * offset[idxs][:, 1]
        x2 = _x2 + ow * offset[idxs][:, 2]
        y2 = _y2 + oh * offset[idxs][:, 3]
        boxes = np.stack((x1, y1, x2, y2, cls[idxs][:, 0]), axis=1)
        '''......'''
        # for idx in idxs:
        #     _box = _rnet_boxes[idx]
        #     _x1 = int(_box[0])
        #     _y1 = int(_box[1])
        #     _x2 = int(_box[2])
        #     _y2 = int(_box[3])
        #
        #     ow = _x2 - _x1
        #     oh = _y2 - _y1
        #
        #     x1 = _x1 + ow * offset[idx][0]
        #     y1 = _y1 + oh * offset[idx][1]
        #     x2 = _x2 + ow * offset[idx][2]
        #     y2 = _y2 + oh * offset[idx][3]
        #     # print('Owangluo',x1, y1, x2, y2)
        #
        #     boxes.append([x1, y1, x2, y2, cls[idx][0]])
        #     # print('置信度：', cls[idx][0])

        boxb = nms_iou.nms(np.array(boxes),  0.4,isMin=True)
        return boxb

    def __pnet_detect(self, image):

        boxes = []
        # boxess =[]

        img = image
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1

        while min_side_len > 12:
            img_data = self.__image_transform(img)
            # print(img_data)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)#输入数据为3维，增加一个批次的维度
            with torch.no_grad():
                _cls, _offest = self.p_net(img_data)
            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data

            idxs = torch.nonzero(torch.gt(cls, 0.6))#取出置信度大于0.6的索引(二维索引）
            # print('置信度',cls[idxs[:, 0], idxs[:, 1]])
            # print(idxs)
            # for idx in idxs:
            #     #idx-->每一个单独的置信度索引，offest-->输出的偏移量，cls[idx[0], idx[1]]-->取出每一个符合条件的置信度，scale-->缩放比例
            #     boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))
            boxes.append(self.__box(idxs, offest, cls[idxs[:,0], idxs[:,1]], scale))

            scale *= 0.708
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = min(_w, _h)
        boxess = torch.cat(boxes).numpy()
        # print(boxess)
        boxb = nms_iou.nms(np.array(boxess), 0.3)

        return boxb

    # 将回归量还原到原图上去
    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):
        #找到建议框的位置
        self._x1 = (start_index[:,1].float() * stride) / scale
        self._y1 = (start_index[:,0].float() * stride) / scale
        self._x2 = (start_index[:,1].float() * stride + side_len) / scale
        self._y2 = (start_index[:,0].float() * stride + side_len) / scale
        # print('建议框：',self._x1,self._y1,self._x2,self._y2)
        ow = self._x2 - self._x1
        oh = self._y2 - self._y1
        #选出置信度达标的索引（二维）拆开成start_index[0]和start_index[1]，再选出对应的偏移量
        _offset = offset[:, start_index[:,0], start_index[:,1]]
        # print('偏移量：',_offset)
        #算出真实框的位置（x1，y1，x2，y2），（_x1，_y1，_x2，_y2）--->建议框

        x1 = self._x1 + ow * _offset[0]
        y1 = self._y1 + oh * _offset[1]
        x2 = self._x2 + ow * _offset[2]
        y2 = self._y2 + oh * _offset[3]
        # print('实际框：',np.stack([x1, y1, x2, y2, cls],axis=1))

        return torch.stack([x1, y1, x2, y2, cls],dim=1)


if __name__ == '__main__':
    # image_file = r'F:\MTCNN\test2\30.jpg'
    # detector = Detector()
    # img_file = r'E:\MTCNN\picture'
    # num = 30
    # with Image.open(image_file) as im:
    #     # im = detector.__image_transform(im)
    #     # im = im.unsqueeze_(0)
    #     # if detector.isCuda:
    #     #     im = im.cuda()
    #     # cls_,_=detector.p_net(im)
    #     # print(cls_)
    #     # boxes = detector.detect(im)
    #     # print("----------------------------")
    #     # boxes = detector.detect(im)
    #     #
    #     boxes = detector.detect(im)
    #     # print(im.size)
    #     imDraw = ImageDraw.Draw(im)
    #     for box in boxes:
    #         x1 = int(box[0])
    #         y1 = int(box[1])
    #         x2 = int(box[2])
    #         y2 = int(box[3])
    #
    #         # print(box[4])
    #         imDraw.rectangle((x1, y1, x2, y2), outline='red',width=2)
    #     im.save('{}/{}.jpg'.format(img_file,num))
    #     im.show()
    '''......'''
    detector = Detector()
    cap = cv2.VideoCapture(r"F:\ycq\video\1.mp4")  # 打开视频文件
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取视频整个帧数
    vid_writer = cv2.VideoWriter(r"F:\ycq\video\wechat_out.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    i = 0
    boxes = []
    boxes_copy = []
    x1 = 0
    non_detected = 0
    # print(cv2.waitKey(1))
    # exit()
    while cv2.waitKey(1) < 0:
        if i == 1:
            start_time = time.time()
        hasFrame, frame = cap.read()  # 重复调用read
        # print(hasFrame,frame)
        # exit()
        if not hasFrame:
            cv2.waitKey(1000)
            cap.release()  # 关闭视频文件（将由析构函数调用，非必须）
            vid_writer.release()
            print("Done processing!")
            break
        frame_rgb = frame[:, :, ::-1]
        frame_img = Image.fromarray(frame_rgb)
        if i % 1 == 0:
            boxes = detector.detect(frame_img)
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        vid_writer.write(frame.astype(np.uint8))
        # cv2.imshow("frame{0}".format(i), frame)
        cv2.imshow("MtCnn detector", frame)
        # if len(boxes) > 1:
        #     print(boxes)
        #     print(boxes_copy)
        #     cv2.waitKey(0)
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
        if i > 0:
            print("fps:", i / (time.time() - start_time))
            print(round((i + 1) * 100 / count, 2), "%")
        i += 1