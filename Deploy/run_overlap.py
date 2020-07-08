import sys
import numpy as np 
import os
import time
import math
from PIL import Image
import cv2
from datetime import datetime
from pynq import Xlnk
from pynq import Overlay
import pynq
import struct
from multiprocessing import Process, Pipe, Queue, Event, Manager
from IoU import Average_IoU

anchor = [1.4940052559648322, 2.3598481287086823, 4.0113013115312155, 5.760873975661669]
bbox_m = [52., 48., 28., 30., 124., 47., 52., 23., 23., 125.]
qm = 131072.0
w = 40
h = 20
IMG_DIR = '../sample1000/'
BATCH_SIZE = 4

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_image_names():
    names_temp = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    names_temp.sort(key= lambda x:int(x[:-4]))
    return names_temp

def get_image_batch():
    image_list = get_image_names()
    batches = list()
    for i in range(0, len(image_list), BATCH_SIZE):
        batches.append(image_list[i:i+BATCH_SIZE])
    return batches

IMAGE_NAMES  = get_image_batch()
IAMGE_NAMES_F = np.array(IMAGE_NAMES).reshape(-1).tolist()
IMAGE_NAMES_LEN = len(IMAGE_NAMES)
IMAGE_BUFF  = []

def stitch(batch):
    image   = np.zeros((4,160,320,4),np.uint8)
    image[0] = np.array(Image.open(IMG_DIR+batch[0]).convert('RGBA').resize((320, 160)))
    image[1] = np.array(Image.open(IMG_DIR+batch[1]).convert('RGBA').resize((320, 160)))
    image[2] = np.array(Image.open(IMG_DIR+batch[2]).convert('RGBA').resize((320, 160)))
    image[3] = np.array(Image.open(IMG_DIR+batch[3]).convert('RGBA').resize((320, 160)))
    return image

def compute_bounding_box(bbox, bbox_origin, batch, result):
    for b in range(4):
        if(bbox_origin[b,4]>0):
            xs = bbox_origin[b][0]*bbox_m[5]/qm
            ys = bbox_origin[b][1]*bbox_m[6]/qm
            ws = bbox_origin[b][2]*bbox_m[7]/qm
            hs = bbox_origin[b][3]*bbox_m[8]/qm
            ws_inb = math.exp(ws)*anchor[2]
            hs_inb = math.exp(hs)*anchor[3]
        else:
            xs = bbox_origin[b][0]*bbox_m[0]/qm
            ys = bbox_origin[b][1]*bbox_m[1]/qm
            ws = bbox_origin[b][2]*bbox_m[2]/qm
            hs = bbox_origin[b][3]*bbox_m[3]/qm
            ws_inb = np.exp(ws)*anchor[0]
            hs_inb = np.exp(hs)*anchor[1]
        xs_inb = sigmoid(xs) + bbox_origin[b][5]
        ys_inb = sigmoid(ys) + bbox_origin[b][6]
        bcx = xs_inb/w
        bcy = ys_inb/h
        bw = ws_inb/w
        bh = hs_inb/h
        bbox[b][0] = bcx - bw/2.0
        bbox[b][1] = bcy - bh/2.0
        bbox[b][2] = bcx + bw/2.0
        bbox[b][3] = bcy + bh/2.0

        x1 = int(round(bbox[b][0] * 640))
        y1 = int(round(bbox[b][1] * 360))
        x2 = int(round(bbox[b][2] * 640))
        y2 = int(round(bbox[b][3] * 360))
        x1 = np.clip(x1,1,640)
        y1 = np.clip(y1,1,360)
        x2 = np.clip(x2,1,640)
        y2 = np.clip(y2,1,360)
        
        result.write(batch[b].split('.')[0].zfill(3) + '.jpg' +' '+str([x1, x2, y1, y2])+'\n')
        print(batch[b], [x1, x2, y1, y2])  

################################## Init FPGA ##################################
xlnk = Xlnk()
xlnk.xlnk_reset()

img   = xlnk.cma_array(shape=[4,160,320,4], dtype=np.uint8)
fm= xlnk.cma_array(shape=(628115*32), dtype=np.uint8)
weight = xlnk.cma_array(shape=(220672),  dtype=np.int16)
biasm  = xlnk.cma_array(shape=(432*16),  dtype=np.int16)
print("Allocating memory done")

parameter = np.fromfile("./weight/SkyNet.bin", dtype=np.int16)
np.copyto(weight, parameter[0:220672])
np.copyto(biasm[0:428*16], parameter[220672:])
print("Parameters loading done")

overlay = Overlay("SkyNet.bit")
print("Bitstream loaded")

SkyNet = overlay.SkyNet
SkyNet.write(0x10, img.physical_address)
SkyNet.write(0x1c, fm.physical_address)
SkyNet.write(0x28, weight.physical_address)
SkyNet.write(0x34, biasm.physical_address)

rails = pynq.get_rails()
recorder = pynq.DataRecorder(rails["power1"].power)

################################## Main process ##################################
bbox_origin = np.empty(64, dtype=np.int16)
bbox  = np.zeros((4,4),dtype=np.float32)
result= open('predict.txt','w+')
batch_buff  = None

image_buff  = np.zeros((4,160,320,4),np.uint8)

print("\n**** Start to detect")
start = time.time()
with recorder.record(0.05):
    for idx, batch in enumerate(IMAGE_NAMES):

        if (idx == 0):
            np.copyto(img, stitch(IMAGE_NAMES[idx]))
        else:
            np.copyto(img, image_buff)

        acc_start = time.time()

        SkyNet.write(0x00, 1)
        if (0 <= idx < IMAGE_NAMES_LEN - 1):
            # pass
            image_buff = stitch(IMAGE_NAMES[idx+1])
        if (0 < idx <= IMAGE_NAMES_LEN - 1):
            compute_bounding_box(bbox ,bbox_origin.reshape(4,16), batch_buff, result)

        isready = SkyNet.read(0x00)
        while( isready == 1 ):
            isready = SkyNet.read(0x00)

        np.copyto(bbox_origin, biasm[428*16:])
        batch_buff = batch

        if (idx == IMAGE_NAMES_LEN - 1):
            compute_bounding_box(bbox ,bbox_origin.reshape(4,16), batch, result)

        acc_end = time.time()
        total_time = acc_end - acc_start

result.close()

end = time.time()
total_time = end - start
print("**** Detection finished\n")
energy = recorder.frame["power1_power"].mean() * total_time
print(str(energy)+'J')

IoU = Average_IoU(IMG_DIR+'ground_truth.txt', 'predict.txt')
print('Total time: ' + str(total_time) + ' s')