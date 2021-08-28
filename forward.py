from options.test_options import BaseOptions
from data.base_dataset import get_transform
from models import create_model
from util.visualizer import save_images
from util.util import tensor2im
from util.face import resize
import numpy as np
import torch
import cv2

from PIL import Image

opt = BaseOptions()
opt.__dict__ = dict(CUT_mode='CUT', checkpoints_dir='./checkpoints', crop_size=200, direction='AtoB', epoch='latest', flip_equivariance=False, gpu_ids=[0], init_gain=0.02, init_type='xavier', input_nc=3, isTrain=False, load_size=256, model='cut', name='painter', nce_idt=True, nce_layers='0,4,8,12,16', netF='mlp_sample', netF_nc=256, netG='resnet_9blocks', ngf=64, no_antialias=False, no_antialias_up=False, no_dropout=True, no_flip=True, normG='instance', output_nc=3, preprocess=[], verbose=False)
model = create_model(opt)
model.setup(opt)
model.parallelize()
model.eval()
transform = get_transform(opt)

def generate(image):
    original = Image.open(image)
    faces = list(resize(np.array(original),opt))
    x = torch.stack([transform(Image.fromarray(face)) for face in faces])
    model.real_A = x
    model.forward()
    output = np.concatenate([tensor2im(image.unsqueeze(0)) for image in model.fake])
    faces = np.concatenate([face for face in faces])
    image = np.concatenate((faces,output),axis=1)
    original = np.array(original)
    if original.shape[0] > image.shape[0]:
        x = int(original.shape[0] / 2 - image.shape[0] / 2)
        image = cv2.copyMakeBorder(image,x,original.shape[0] - image.shape[0] - x,0,0,cv2.BORDER_CONSTANT)
    else:
        original = cv2.resize(original,(int(original.shape[1] * image.shape[0] / original.shape[0]),image.shape[0]))
    image = np.concatenate((original,image),axis=1)
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    result = generate("test.jpg")
    cv2.imwrite("output.png",result)
