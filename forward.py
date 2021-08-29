from data.base_dataset import get_transform
from utils.general import create_folder
from models import create_model

from util.face import resize, init
from util.util import tensor2im

from glob import glob
from PIL import Image
import numpy as np
import torch
import cv2
import os

from argparse import ArgumentParser

parser = ArgumentParser(description="Painter")

parser.add_argument("--verbose",action="store_true",help="print extra messages")
parser.add_argument("--gpu-ids","--gpus",nargs="+",default=[0],type=int,help="GPUs to run on")
parser.add_argument("--min-face-size","--size",default=50,type=int,help="minimum width of detected face (px)")
subparsers = parser.add_subparsers()

image = subparsers.add_parser("image")
image.add_argument("--save-path",default="output.png",help="output save location")
image.add_argument("--input-image",default="example.png",required=True,help="input image location")
image.set_defaults(action="image")

folder = subparsers.add_parser("folder")
folder.add_argument("--output-folder",default="output/",help="output save location")
folder.add_argument("--input-folder",default=".",required=True,help="input image location")
folder.add_argument("--extensions",nargs="+",default=["png", "jpeg", "jpg", "bmp", "tiff", "tif"],help="image file extensions")
folder.set_defaults(action="folder")
parser.set_defaults(CUT_mode='CUT', checkpoints_dir='./checkpoints', crop_size=200, direction='AtoB', epoch='latest', flip_equivariance=False, init_gain=0.02, init_type='xavier', input_nc=3, isTrain=False, load_size=256, model='cut', name='painter', nce_idt=True, nce_layers='0,4,8,12,16', netF='mlp_sample', netF_nc=256, netG='resnet_9blocks', ngf=64, no_antialias=False, no_antialias_up=False, no_dropout=True, no_flip=True, normG='instance', output_nc=3, preprocess=[])

args = parser.parse_args()
init(args.gpu_ids[0])
model = create_model(args)
model.setup(args)
model.parallelize()
model.eval()
transform = get_transform(args)

def generate(image,opt):
    print(image)
    original = Image.open(image)
    if opt.verbose:
        print("Extracting faces")
    faces = list(resize(np.array(original),opt))
    if len(faces) == 0:
        print("No faces detected")
        return cv2.cvtColor(np.array(original),cv2.COLOR_BGR2RGB)
    x = torch.stack([transform(Image.fromarray(face)) for face in faces])
    model.real_A = x
    if opt.verbose:
        print("Generating fake samples")
    model.forward()
    if opt.verbose:
        print("Collating results")
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
    if args.action == "image":
        result = generate(args.input_image,args)
        create_folder(args.save_path)
        cv2.imwrite(args.save_path,result)
    elif args.action == "folder":
        folder = args.input_folder
        if not folder.endswith("/") and not folder.endswith("\\"):
            folder += "/"
        for extension in args.extensions:
            for image in glob(f"{folder}**/*.{extension}",recursive=True):
                result = generate(image,args)
                save_path = os.path.join(args.output_folder,os.path.relpath(image,folder))
                create_folder(save_path)
                cv2.imwrite(save_path,result)
