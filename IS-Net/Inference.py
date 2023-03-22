import os
import time
import numpy as np
from skimage import io
import time
from glob import glob
from tqdm import tqdm

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models import *
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # changelleng dataset:

    dataset_path="./test_data"  #Your dataset path
    checkpoint = torch.load("./saved_models/DIS_model.pth")
    result_path="./test_data"  #The folder path that you want to save the results
    input_size=[1024,1024]
    net_real=ISNetDIS()
    net_syn = ISNetDIS()

    if torch.cuda.is_available():
        net_real.load_state_dict(checkpoint['model_path_real'])
        net_real=net_real.cuda()
        net_syn.load_state_dict(checkpoint['model_path_syn'])
        net_syn=net_syn.cuda()
    else:
        print('torch.cuda is not available')
    net_real.eval()
    net_syn.eval()

    im_list = glob(dataset_path+"/*.src.jpg")

    with torch.no_grad():
        print('produce prospects masks')
        for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
            im_name=im_path.split('/')[-1].split('.')[0]
            im = io.imread(im_path)
            if len(im.shape) < 3:
                im = im[:, :, np.newaxis]
            im_shp=im.shape[0:2]
            im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
            im_tensor = F.upsample(torch.unsqueeze(im_tensor,0), input_size, mode="bilinear").type(torch.uint8)
            image = torch.divide(im_tensor,255.0)
            image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])

            if torch.cuda.is_available():
                image=image.cuda()
            if im_name[:4] == 'real':
                result=net_real(image)
            else:
                result=net_syn(image)
            result=torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
            ma = torch.max(result)
            mi = torch.min(result)
            result = (result-mi)/(ma-mi)
            result = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
            io.imsave(os.path.join(result_path,im_name+".mask_src.png"),result)
