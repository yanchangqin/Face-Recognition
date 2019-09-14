import numpy as np

def iou(box,boxes,isMIN = False):
    #计算各个框的面积
    box_areas = (box[2]-box[0])*(box[3]-box[1])
    boxes_areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    #找交集
    # print(boxes[:,0])
    intersection_x1 = np.maximum(box[0],boxes[:,0])
    intersection_y1 = np.maximum(box[1], boxes[:,1])
    intersection_x2 = np.minimum(box[2], boxes[:, 2])
    intersection_y2 = np.minimum(box[3], boxes[:, 3])
    #判断是否有交集
    w = np.maximum(0,(intersection_x2-intersection_x1))
    h = np.maximum(0,(intersection_y2-intersection_y1))
    #计算交集的面积

    intersection_areas= w*h

    if isMIN ==True:
        result =np.divide(intersection_areas,np.minimum(box_areas,boxes_areas))
    else:
        result = np.divide(intersection_areas,box_areas+boxes_areas-intersection_areas)
    return result

def nms(boxes,para = 0.3,isMIN=False):
    # print("P网络的框：",boxes)
    conf_sort = boxes[-(boxes[:,4]).argsort()]
    boxes_rest = []
    while conf_sort.shape[0]>1:
        a_box = conf_sort[0]
        boxes_rest.append(a_box)
        b_box = conf_sort[1:]
        # print(b_box)
        index = np.where(iou(a_box,b_box,isMIN)<para)
        conf_sort = b_box[index]
        # print(box_conf)
    if conf_sort.shape[0]>0:
        boxes_rest.append(conf_sort[0])
    return np.stack(boxes_rest)

def convert_to_square(boxb):
    square_box =boxb.copy()
    # print(square_box)
    x1 = boxb[:,0]
    y1 = boxb[:,1]
    x2 = boxb[:,2]
    y2 = boxb[:,3]
    w = x2-x1
    h = y2-y1
    max_side = np.maximum(w,h)

    square_box[:,0] = x1+w/2-max_side/2
    square_box[:,1] = y1+h/2-max_side/2
    square_box[:,2] = square_box[:,0]+max_side
    square_box[:,3] = square_box[:,1]+max_side
    # print(square_box)
    return square_box



# if __name__ == '__main__':
#     a = np.array([1, 1, 12, 12])
# bs = np.array([[2,2,3,3,5],[4,4,5,5,8],[18,18,27,27,9],[8,8,7,7,6]])
    # iou(a, bs, isMIN=False)
    # print(nms(bs))
# convert_to_square(bs)
