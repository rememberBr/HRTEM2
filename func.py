import numpy as np
import cv2
import pdb
import os

from unet3080 import *

from skimage import measure
import math
import copy

import itertools
import json
import re

class PDF():
    def __init__(self):
        self.colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (0,0,127), (0,127,0), (127,0,0), (0,127,127), (127,127,0), (127,0,127)]
        self.PDFdata = {}
        self.PDFlist = []
        self.matchlist = []
        self.color = {}
        self.colorid = 0
        self.res = {}

    def readPDF(self, path):
        tmp = []
        with open(path, "r") as f:
            need = False
            for line in f.readlines():
                if line[:4] == 'd(A)':
                    need = True
                if need:
                    line = line.strip('\n')
                    tmp.append([i for i in line.split(' ') if i != ''])
        new = [[i[0], i[3],i[4],i[5]] for i in tmp[1:]] 
        return new


    def angle(self, x, y):
        x = np.array(x)
        y = np.array(y)

        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))

        cos = x.dot(y)/(Lx*Ly)
        ang = round(np.arccos(cos)*360/2/np.pi, 2)
        return ang

    def loadPDF(self):

        for root, non, files in os.walk('PDF/'):
            pass
        for obj in files:
            path = 'PDF/'+obj
            reader = self.readPDF(path)
            dic = {i[0]:i[1:] for i in reader}
            id = [i[0] for i in reader]
            combination = []
            for i in itertools.permutations(id, 2):
                combination.append(i)
            newData = []
            for i in range(len(combination)):
                x = [int(j) for j in dic[combination[i][0]]]
                y = [int(j) for j in dic[combination[i][1]]]
                ang = self.angle(x, y)
                if i < len(id):
                    newData.append([combination[i], ang, id[i], np.cross(x,y)])
                else:
                    newData.append([combination[i], ang, '', np.cross(x,y)])
            name = re.split(r'[\\/]', path.split('.')[0])[-1]
            self.PDFdata[name] = newData  
            self.PDFlist.append(name)
   
    def chosePDF(self, Crystals, name):
        colorid = 0
        paths = name.split(' ')
        for i in paths:
            Crystals.args.PDFs[i] = self.PDFdata[i]
            Crystals.args.matchlist.append(i)
            Crystals.args.color[i] = self.colors[colorid]
            colorid += 1
        return 1



def load_unet():
    myunet = myUnet()
    model = myunet.get_unet_lca()
    model.load_weights('lcaunet.h5')
    model.predict(np.ones((1,64,64,1))/2, verbose=1)
    return model

def imgSplit(args, img, grid_h, step):
    # pdb.set_trace()
    args.x1 = 0
    args.y1 = 0
    args.x2 = img.shape[1]
    args.y2 = img.shape[0]
    w = args.x2-args.x1
    h = args.y2-args.y1
    if h % grid_h != 0:
        new_y = args.y2 + (grid_h - h % grid_h)
        
        if new_y > img.shape[0]:
            new_y = args.y1 - (grid_h - h % grid_h)
            if new_y<0:
                args.y1 += (h % grid_h)
            else:
                args.y1 -= (grid_h - h % grid_h)
        else:
            args.y2 += (grid_h - h % grid_h)
    if w % grid_h != 0:
        new_x = args.x2 + (grid_h - w % grid_h)
        if new_x > img.shape[0]:
            new_x = args.x1 - (grid_h - w % grid_h)
            if new_x<0:
                args.x1 += (w % grid_h)
            else:
                args.x1 -= (grid_h - w % grid_h)
        else:
            args.x2 += (grid_h - w % grid_h)
    num_w = (((args.x2-args.x1-grid_h)//step)+1)
    num_h = (((args.y2-args.y1-grid_h)//step)+1)
    return num_w, num_h, num_w*num_h 
        
def load_data(FFTs):
    imgs_pre = np.ndarray((len(FFTs), 64, 64, 1), dtype=np.float32)
    for i in range(len(FFTs)):
        im = FFTs[i].reshape((64, 64, 1)).astype('float32')
        imgs_pre[i] = (im/255)**1*255 
        # imgs_pre[i] = im
    return imgs_pre.astype('uint8')


#计算夹角
def angle(v1, v2):
    dx1 = v1[0]
    dy1 = v1[1]
    dx2 = v2[0]
    dy2 = v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = angle1 * 180/math.pi
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = angle2 * 180/math.pi
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

#约等于
def Aequal(num1, num2, dist_error, angle_error):
    if type(num1) == np.ndarray or type(num1) == list:
        if abs(num1[0]-num2[0])<dist_error and abs(num1[1]-num2[1])<dist_error:
            return True
        else:
            return False
    else:
        if abs(num1-num2)<angle_error:
            return True
        else:
            return False

#计算点距离 
def Distance(point1,point2):
    vec1 = np.array(point1)
    vec2 = np.array(point2)
    distance = np.linalg.norm(vec1 - vec2)
    return distance

#判断是否属于同一条直线
def isBoomerang(points, dist_error, angle_error):
    x1, y1 = points[0][0], points[0][1]
    x2, y2 = points[1][0], points[1][1]
    x3, y3 = points[2][0], points[2][1]
    return Aequal((x3 - x1) * (y2 - y1), (x2 - x1) * (y3 - y1), dist_error, angle_error)

#重心法求斑点中心
def point_center(img, image_np, name):
    img = (255-img*255)/255
    mask = img>0.5
    label_image, num = measure.label(mask, connectivity = 1, return_num = True)
    label = {i:[] for i in range(1, num+1)}
    for i in range(1, num+1):
        x_y = np.where(label_image == i)
        label[i].append(list(zip(x_y[0], x_y[1])))                
    vecter = []

    for i in range(1,num+1):
        tmp = np.array(label[i][0])
        x = tmp[:,0]
        y = tmp[:,1]
        imtmp = []
        for j in tmp:
            imtmp.append(img[j[0],j[1]])
        w1 = imtmp/sum(imtmp)
        aa = np.matrix(w1)
        w2 = aa*tmp

        if abs(w2[0, 1] - 32) < 1 and abs(w2[0, 0] - 32) < 1: #去除中心点
            continue
        else:
            vecter.append([w2[0, 1] - 32, w2[0, 0] - 32])
    vecter = np.array(vecter)
    return vecter

#去除关于中心对称的点
def clear_symmetry(vecter, image_np, name, dist_error, angle_error):
    del_idx = []
    
    for num in range(len(vecter)):
        sta = False #是否存在对称点
        for num1 in range(num+1,len(vecter)):
            if Aequal(vecter[num], -vecter[num1], dist_error, angle_error):
                sta = True
                if vecter[num][1]<=vecter[num1][1]:
                    del_idx.append(num1)
                else:
                    del_idx.append(num)
        #无对称也删除  模型质量非常好不一定需要这个！！！，看情况是否添加
        if not sta:
            del_idx.append(num)
    del_idx = list(set(del_idx))
    del_idx.sort(reverse=True)
    for d in del_idx:
        vecter = np.delete(vecter, d, axis=0)  
    return vecter

#清除延长线上距离中心最远的点
def clear_extension_cord(vecter, image_np, name, dist_error, angle_error):
    del_idx = []
    for num in range(len(vecter)):
        for num1 in range(num+1,len(vecter)):
            if isBoomerang([[32, 32], [vecter[num, 0]+32, vecter[num, 1]+32], [vecter[num1][0]+32, vecter[num1][1]+32]], dist_error, angle_error):
                if np.linalg.norm(vecter[num]) <= np.linalg.norm(vecter[num1]):
                    del_idx.append(num1)
                else:
                    del_idx.append(num)
    del_idx = list(set(del_idx))
    del_idx.sort(reverse=True)

    for d in del_idx:
        vecter = np.delete(vecter, d, axis=0)
    return vecter

#寻找最终存在的平行四边形的数据
def parallelogram_data(magnification, pre_res, PDF_data, dist_error, angle_error, PDF_dist_error, PDF_angle_error):
    all_data = {}
    name = 0
    dists = {}
    angs = {}
    orientations = {}
    # pdb.set_trace()
    # 下面这点东西是通用的，也许可以抽出来，不需要每次都处理pdf卡片
    for k, v in PDF_data.items():
        dists[k] = [i[2] for i in v if i[2] != '']
        angs[k] = {i[0]:i[1] for i in v}
        orientations[k] = {i[0]:i[3] for i in v}
    # pdb.set_trace()
    for im in pre_res:
        # pdb.set_trace()
        im = im.reshape(64,64)
        image_np = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        vecter = point_center(im, image_np, name)  
        #寻找关于中心的对称的斑点，若存在，删除靠下面的一个点
        # pdb.set_trace()
        vecter = clear_symmetry(vecter, image_np, name, dist_error, angle_error)
        #清除延长线上距离中心最远的点
        vecter = clear_extension_cord(vecter, image_np, name, dist_error, angle_error)
        minDists = {k:[] for k, v in dists.items()}
        if len(vecter) <2:
            pass
        else:
            dist = [[10/(np.linalg.norm(i)*magnification),i] for i in vecter] #计算每个点到中心的距离 
            for i in dist:
                for ke, va in dists.items():
                    dis = [[j,i[1]] for j in va if (1+PDF_dist_error)*float(j)>i[0] and (1-PDF_dist_error)*float(j)<i[0]]
                    if dis:
                        minDists[ke].extend(dis)
        all_data[name] = copy.deepcopy(minDists)
        name += 1
    tmp = []
    # pdb.set_trace()
    for k, v in all_data.items():
        for ke, va in v.items(): 
            if len(va)>=2:
                tmp.append([ke, va, k])
    # pdb.set_trace()
    res = {}
    for i in tmp:
        key = i[2]
        comb = [j for j in itertools.combinations(i[1], 2)]
        comb1 = [(j[0][0],j[1][0],str(orientations[i[0]][(j[0][0],j[1][0])])) for j in comb if j[0][0]!=j[1][0]\
                and angs[i[0]][(j[0][0],j[1][0])]*(1 + PDF_angle_error)>angle(j[0][1],j[1][1]) \
                and angs[i[0]][(j[0][0],j[1][0])]*(1 - PDF_angle_error)<angle(j[0][1],j[1][1])]    
        if comb1 !=[]:
            try:
                res[key][i[0]] = comb1
            except KeyError:
                res[key] = {i[0]:comb1}
    # pdb.set_trace()
    return res

def show(loadings, img, step):
    loadings = cv2.resize(loadings, img.shape)*255
    loadings = loadings.astype(np.uint8)

    _, thresh = cv2.threshold(loadings, 150, 255, cv2.THRESH_BINARY) 
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # 获取最大连通域的统计数据
    sizes = stats[1:, -1]  # 去掉背景的 0 标签
    max_label = 1 + np.argmax(sizes)  # 最大连通域的标签
   
    # 创建一个黑色背景图像
    mask = np.zeros_like(thresh)

    # 在黑色背景图像上绘制最大连通域
    mask[output == max_label] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.cv2.CHAIN_APPROX_SIMPLE) 
    for i in contours[0]:
        if i[0,0]!=0:
            i[0,0] += step
        if i[0,1]!=0:
            i[0,1] += step
    img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img, contours, -1, (0, 255, 255), 5)

    return img
