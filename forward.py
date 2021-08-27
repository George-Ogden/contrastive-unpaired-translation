from options.test_options import BaseOptions
from data.base_dataset import get_transform
from models import create_model
from util.visualizer import save_images
from util.util import tensor2im

from PIL import Image

opt = BaseOptions()
opt.__dict__ = dict(CUT_mode='CUT', checkpoints_dir='./checkpoints', crop_size=200, direction='AtoB', epoch='latest', flip_equivariance=False, gpu_ids=[0], init_gain=0.02, init_type='xavier', input_nc=3, isTrain=False, load_size=256, model='cut', name='painter', nce_idt=True, nce_layers='0,4,8,12,16', netF='mlp_sample', netF_nc=256, netG='resnet_9blocks', ngf=64, no_antialias=False, no_antialias_up=False, no_dropout=True, no_flip=True, normG='instance', output_nc=3, preprocess=[], verbose=False)
model = create_model(opt)
model.setup(opt)
model.parallelize()
model.eval()
if __name__ == '__main__':
    transform = get_transform(opt)
    x = Image.open("test.png")
    x = transform(x).unsqueeze(0)
    model.real_A = x
    model.forward()
    y = tensor2im(model.fake)
    y = Image.fromarray(y)
    y.save("output.png")