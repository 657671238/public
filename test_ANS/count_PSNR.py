import torch.nn as nn
import numpy as np
import cv2
import argparse
from PIL import Image


import pytorch_ssim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, utils

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import logging
from datetime import datetime
import os
import numpy
import math
import random

parser = argparse.ArgumentParser()

parser.add_argument('--target',  default='init', help='save result')
#parser.add_argument('--target',  default='LOL_test/high', help='save result')
parser.add_argument('--answer',  default='nlm', help='save result')
# Bilateral_filter  gaussian  guided_filter  non_local_means
opt = parser.parse_args()

def calc_ssim(img1,img2):
    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=False)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value


def calc_psnr(img1, img2):
    mse = np.mean((img1 * 255. - img2 * 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 255
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


list1 = os.listdir(opt.answer)
list2 = os.listdir(opt.target)
list1.sort(key=lambda x: int(x[:-4]))
list2.sort(key=lambda x: int(x[:-4]))
images = [os.path.join(opt.answer, img) for img in list1]
noises = [os.path.join(opt.target, img) for img in list2]

transform = transforms.Compose([
    transforms.ToTensor()
])
lengh = len(images)

ssim = np.zeros(lengh)
psnr = np.zeros(lengh)
for idx in range(lengh):

    image, noise = images[idx], noises[idx]
    image = cv2.imread(image)
    noise = cv2.imread(noise)
    image = transform(np.asarray(image)).float()
    noise = transform(np.asarray(noise)).float()

    psnr[idx] = calc_psnr(image.detach().cpu().numpy(), noise.detach().cpu().numpy())
    ssim[idx] = calc_ssim(image.detach().unsqueeze(0).cpu(), noise.detach().unsqueeze(0).cpu())
    print(idx, ssim[idx],psnr[idx])
print('cals: psnr %.6f ssim %.6f'%(psnr.mean(),ssim.mean()))
