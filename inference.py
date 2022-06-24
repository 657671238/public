
from __future__ import print_function
import os
import time
import socket
import pandas as pd
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from CURTransformer import CURTransformer
from data import get_training_set, get_eval_set
from skimage.measure.simple_metrics import compare_psnr
from dataloader_inf import datset
import cv2
import math
import pytorch_ssim

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate. default=0.0001')
parser.add_argument('--data_augmentation', type=bool, default=False, help='if adopt augmentation when training')
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR', help='the training dataset')
parser.add_argument('--Ispretrained', type=bool, default=False, help='If load checkpoint model')
parser.add_argument('--pretrained_sr', default='CURTransformernet_epoch_765_.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', default='./checkpoint/', help='Location to load checkpoint models')
parser.add_argument("--noiseL", type=float, default=50, help='noise level')
parser.add_argument('--save_folder', default='./checkpoint/', help='Location to save checkpoint models')
parser.add_argument('--statistics', default='./statistics/', help='Location to save statistics')

# Testing settings
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size, default=1')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--test_dataset', type=str, default='Set12', help='the testing dataset')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')

# Global settings
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
parser.add_argument('--data_dir', type=str, default='/home/dongxuan/users/xukang/SIDD_Medium_Srgb', help='the dataset dir')
parser.add_argument('--model_type', type=str, default='CURTransformer', help='the name of model')
parser.add_argument('--Isreal', default=True, help='If training/testing on RGB images')
parser.add_argument('--csvfile', type=str, default='hello.csv', help='csv_files')


parser.add_argument('--root_dir',  default='images/our485', help='save result')
parser.add_argument('--sigma',  default=70, type=int, help='save result')
parser.add_argument('--batchsize',  default=1, type=int, help='save result')
opt = parser.parse_args()

def modcrop(image, modulo):
    h, w = image.shape[0], image.shape[1]
    h = h - h % modulo
    w = w - w % modulo
    return image[:h, :w]

def calc_rmse(a, b):
    a = a * 255
    b = b * 255
    return np.sqrt(np.mean(np.power(a - b, 2)))

def calc_psnr(img1, img2):
    mse = np.mean((img1 * 255. - img2 * 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 255
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_ssim(img1,img2):
    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=False)
    ssim_value = pytorch_ssim.ssim(img1, img2).item()
    return ssim_value


net = CURTransformer(
        hidden_dim=32,
        layers=(2, 2, 6, 2),
        heads=1,
        channels=32,
        num_classes=3,
        head_dim=32,
        window_size=14,
        downscaling_factors=1,
        relative_pos_embedding=True
    )
net = torch.nn.DataParallel(net, device_ids=range(opt.gpus))
model_name = os.path.join(opt.pretrained, opt.pretrained_sr)
net.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
count = 0
if not os.path.exists('./images'):
    os.mkdir('./images')
@torch.no_grad()
def validate(net):
    test_dataset = datset(opt.data_dir)
    batchsize = 1
    dataloader_test = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=False, num_workers=0, drop_last=True)
    net.eval()
    global count
    length = len(dataloader_test)
    rmse = np.zeros(length)
    psnr = np.zeros(length)
    ssim = np.zeros(length)
    for idx, data in enumerate(dataloader_test):
        gt, target = data['gt'], data['target'].cuda()
        b,c,h,w = gt.shape
        new_h,new_w = h//224*224+224, w//224*224+224
        #扩展后大图
        new_out = torch.zeros(b,c,h,w)
        for i in range(2):
            for j in range(2): 
                new_out[:,:,h*(i)//2:h*(i+1)//2,w*(j)//2:w*(j+1)//2] = net(target[:,:,h*(i)//2:h*(i+1)//2,w*(j)//2:w*(j+1)//2].cuda())
        out = new_out
        #out1 = cv2.resize(out[0].cpu().permute(1, 2, 0).numpy(), (600,400))
        cv2.imwrite('./images/{}.png'.format(count), out[0].cpu().permute(1, 2, 0).numpy() * 255)
        count = count + 1
        # if idx%300==0:
        #     cv2.imwrite('./images/SR/{}_gt.png'.format(count), gt[0].cpu().permute(1, 2, 0).numpy() * 255)
        #     cv2.imwrite('./images/SR/{}_target.png'.format(count), target[0].cpu().permute(1, 2, 0).numpy() * 255)
        #     cv2.imwrite('./images/SR/{}_ans.png'.format(count), out[0].cpu().permute(1, 2, 0).numpy() * 255)
        #     count = count + 1
        calc_p = 0
        calc_r = 0
        calc_s = 0
        for i in range(opt.batchsize):
            calc_r += calc_rmse(gt[i].cpu().numpy(), out[i].detach().cpu().numpy())
            calc_p += calc_psnr(gt[i].cpu().numpy(), out[i].detach().cpu().numpy())
            calc_s += calc_ssim(gt[i].unsqueeze(0).cpu(), out[i].unsqueeze(0).cpu())
        rmse[idx] = calc_r/opt.batchsize
        psnr[idx] = calc_p/opt.batchsize
        ssim[idx] = calc_s/opt.batchsize
        print('%d:rmse:%f psnr:%f ssim:%f'%(idx, rmse[idx], psnr[idx], ssim[idx]))
    return rmse, psnr, ssim

rmse,psnr,ssim = validate(net)
print('mean_rmse:%f mean_psnr:%f mean_ssim:%f'%(rmse.mean(), psnr.mean(), ssim.mean()))
