import os
from PIL import Image
import numpy as np
import iou_nms
import traceback
import PIL.ImageDraw as draw

anno_src = r"F:\MTCNN\data\Anno\list_bbox_celeba.txt"
img_dir = r"F:\MTCNN\data\img_celeba"
save_path = r"E:\MTCNN\celeba_new"

for face_size in [12,24,48]:
    positive_image_dir = os.path.join(save_path,str(face_size),'positive')
    negative_image_dir = os.path.join(save_path, str(face_size), 'negative')
    part_image_dir = os.path.join(save_path, str(face_size), 'part')

    for path in [positive_image_dir,negative_image_dir,part_image_dir] :
        if not os.path.exists(path) :
            os.makedirs(path)

    positive_anno_filename = os.path.join(save_path,str(face_size),'positive.txt')
    negative_anno_filename = os.path.join(save_path, str(face_size), 'negative.txt')
    part_anno_filename = os.path.join(save_path, str(face_size), 'part.txt')

    positive_count = 0
    negative_count = 0
    part_count = 0

    try:
        positive_anno_dir = open(positive_anno_filename,'w')
        negative_anno_dir = open(negative_anno_filename,'w')
        part_anno_dir = open(part_anno_filename,'w')

        for i, line in enumerate(open(anno_src)):
            if i < 2:
                continue
            try:
                lst = line.split()
                image_filename = lst[0].strip()
                image_dir = os.path.join(img_dir,image_filename)
                with Image.open(image_dir) as img:

                    x1 = int(lst[1])
                    y1 = int(lst[2])
                    w = int(lst[3])
                    h = int(lst[4])
                    x2 = x1+w
                    y2 = y1+h
                    #修正后的标签框
                    x1_fix = x1 + 10
                    y1_fix = y1 + 5
                    x2_fix= x2 - 10
                    y2_fix= y2 - 10

                    box = [[x1_fix,y1_fix,x2_fix,y2_fix]]

                    x_center =x1_fix+(x2_fix-x1_fix)/2
                    y_center = y1_fix+(y2_fix-y1_fix)/2

                    w_fix = x2_fix - x1_fix
                    h_fix = y2_fix-y1_fix

                    for i in range(4):
                        w_ = np.random.randint(-w*0.3,0.3*w)
                        h_ = np.random.randint(-h * 0.3,0.3 * h)
                        #偏移后的中心点
                        x_c = x_center+w_
                        y_c = y_center+h_
                        #偏移后的正方形框
                        side_lenp = np.random.randint(int(min(w_fix, h_fix) * 0.8), np.ceil(1.25 * max(w_fix, h_fix)))#np.ceil表示选择>=该数的最小整数
                        x1_ = np.max(x_c - side_lenp / 2, 0)
                        y1_ = np.max(y_c - side_lenp / 2, 0)
                        x2_ = x1_ + side_lenp
                        y2_ = y1_ + side_lenp

                        crop_box_p = np.array([x1_,y1_,x2_,y2_])
                        #计算偏移量
                        offset_x1 = (x1_fix - x1_) / side_lenp
                        offset_y1 = (y1_fix - y1_) / side_lenp
                        offset_x2 = (x2_fix - x2_) / side_lenp
                        offset_y2 = (y2_fix - y2_) / side_lenp

                        face_crop = img.crop(crop_box_p)
                        face_resize = face_crop.resize((face_size,face_size),Image.ANTIALIAS)
                        #计算IOU
                        c_iou = iou_nms.iou(crop_box_p,np.array(box))[0]

                        if c_iou>0.65:
                            positive_anno_dir.write('positive/{0}.jpg {1} {2} {3} {4} {5} \n'.format(positive_count,1,offset_x1,offset_y1,offset_x2,offset_y2))
                            positive_anno_dir.flush()
                            # face_resize.save(os.path.join(positive_image_dir,'{}.jpg'.format(positive_count)))
                            positive_count+=1

                        elif  0.5>c_iou>0.3:
                            part_anno_dir.write('part/{0}.jpg {1} {2} {3} {4} {5} \n'.format(part_count,2,offset_x1,offset_y1,offset_x2,offset_y2))
                            part_anno_dir.flush()
                            # face_resize.save(os.path.join(part_image_dir,'{}.jpg'.format(part_count)))
                            part_count+=1
                        elif c_iou<0.2:
                            negative_anno_dir.write(
                                'negative/{0}.jpg {1} {2} {3} {4} {5} \n'.format(negative_count, 0, 0, 0, 0, 0))
                            negative_anno_dir.flush()
                            face_resize.save(os.path.join(negative_image_dir, '{}.jpg'.format(negative_count)))
                            negative_count += 1
                    for i in range(3):
                        w_o, h_o = img.size
                        if min(w_o, h_o - y2)<12:
                            pass
                        else:
                            side_len1 = np.random.randint(12, min(w_o, h_o - y2))
                            x1_negative = np.random.randint(0, w_o - side_len1)
                            y1_negative = np.random.randint(y2, h_o - side_len1)
                            x2_negative = x1_negative + side_len1
                            y2_negative = y1_negative + side_len1
                            i_ = x2_negative - x1_negative
                            j_ = y2_negative - y1_negative
                            if i_>0 and j_>0:
                                crop_box1 = np.array([x1_negative,y1_negative,x2_negative,y2_negative])
                                face_crop = img.crop(crop_box1)
                                face_resize = face_crop.resize((face_size, face_size),Image.ANTIALIAS)
                                negative_anno_dir.write(
                                    'negative/{0}.jpg {1} {2} {3} {4} {5} \n'.format(negative_count, 0, 0, 0, 0, 0))
                                negative_anno_dir.flush()
                                face_resize.save(os.path.join(negative_image_dir, '{}.jpg'.format(negative_count)))
                                negative_count += 1


            except Exception as e:
                 traceback.print_exc()
    finally:
        positive_anno_dir.close()
        negative_anno_dir.close()
        part_anno_dir.close()



