import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

model = None
device = None
def init(gpu):
    # intialise model
    global model, device
    device = select_device(str(gpu))
    model = attempt_load("checkpoints/face/yolo.pt", map_location=device)
    model.half()
    model(torch.zeros(1, 3, 512, 512).to(device).type_as(next(model.parameters())))

def xywh(im0):
    """
    returns x,y,w,h
    """
    # format image
    stride = int(model.stride.max())
    img = letterbox(im0, 512,stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half()
    img /= 255.0
    if img.ndimension() == 3: img = img.unsqueeze(0)
    # generate prodictions
    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.5, 0.45, agnostic=True)
    if len(pred):
        det = pred[0]
        # resize predictions
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            x1, y1, x2, y2 = xyxy
            # return each prediction and x,y,w,h
            yield int((x1+x2)/2), int((y1+y2)/2), int(x2-x1), int(y2-y1)

def crop(image,opt):
    # centre crop image
    x = int(opt.load_size / 2 - opt.crop_size / 2)
    h = opt.crop_size
    return image[x:x+h,x:x+h]

def resize(image,opt):
    shape = [opt.load_size] * 2
    # values taken from initial dataset
    width = 86
    height = 119
    for x,y,w,h in xywh(image):
        if w < opt.min_face_size:
            # ignore small faces
            continue
        ratio = np.sqrt(float(width*height)/float(w*h))
        # resize image using linear algebra
        yield crop(cv2.warpAffine(image,np.float32([[ratio,0,int(shape[0]/2-ratio*x)],[0,ratio,int(shape[1]/2-ratio*y)]]),(shape[0],shape[1])),opt)
        
