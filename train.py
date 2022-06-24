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
from dataloader_SIDD import datset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
parser.add_argument('--pretrained_sr', default='noise25.pth', help='sr pretrained base model')
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

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        # target = Variable(batch)
        # noise = torch.FloatTensor(target.size()).normal_(mean=0, std=opt.val_noiseL / 255.)
        # input = target + noise

        input = batch['target'].cuda()
        target = batch['gt'].cuda()

        model.zero_grad()
        optimizer.zero_grad()
        t0 = time.time()

        prediction = model(input)

        # Corresponds to the Optimized Scheme
        loss = criterion(prediction, target)/(input.size()[0]*2)

        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        # if (epoch+1) % 10 == 0:
        #     model.eval()
        #     SC = 'net_epoch_' + str(epoch) + '_' + str(iteration + 1) + '.pth'
        #     torch.save(model.state_dict(), os.path.join(opt.save_folder, SC))
        #     model.train()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def test(testing_data_loader):
    psnr_test= 0
    model.eval()
    for batch in testing_data_loader:
        input = batch['target'].cuda()
        target = batch['gt'].cuda()
        with torch.no_grad():
            prediction = model(input)
            prediction = torch.clamp(prediction, 0., 1.)
        psnr_test += batch_PSNR(prediction, target, 1.)
    print("===> Avg. PSNR: {:.4f} dB".format(psnr_test / len(testing_data_loader)))
    return psnr_test / len(testing_data_loader)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch,psnr):
    model_out_path = opt.save_folder+hostname+opt.model_type+"_psnr_{}".format(psnr)+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    print('===> Loading datasets')

    if opt.Isreal:
        train_set = datset(opt.data_dir, train=True)
        training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=4,drop_last=True)
        test_set = datset(opt.data_dir, train=False)
        testing_data_loader = DataLoader(dataset=test_set, batch_size=opt.testBatchSize, shuffle=False, num_workers=0, drop_last=True)
    else:
        train_set = Dataset_from_read(os.path.join(opt.data_dir, 'train'), opt.hr_train_dataset, opt.upscale_factor,
                                     opt.patch_size, opt.data_augmentation)
        training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

        test_set = Dataset_from_read(os.path.join(opt.data_dir+'test', opt.test_dataset), opt.upscale_factor)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    print('===> Building model ', opt.model_type)
    model = CURTransformer(
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

    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    criterion = nn.L1Loss()

    print('---------- Networks architecture -------------')
    print_network(model)
    print('----------------------------------------------')

    if opt.Ispretrained:
        model_name = os.path.join(opt.pretrained, opt.pretrained_sr)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print(model_name + ' model is loaded.')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    min_psnr = -1
    PSNR = []
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        train(epoch)
        psnr = test(testing_data_loader)
        PSNR.append(psnr)
        data_frame = pd.DataFrame(
            data={'epoch': epoch, 'PSNR': PSNR}, index=range(opt.start_iter, epoch+1)
        )
        data_frame.to_csv(os.path.join(opt.statistics, opt.model_type + opt.csvfile), index_label='index')
        # learning rate is decayed by a factor of 10 every half of total epochs
        if (epoch + 1) % (opt.nEpochs / 2) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10.0
            print('Learning rate decay: lr={}'.format(param_group['lr']))
        if min_psnr==-1:
            min_psnr = psnr
        else:
            if psnr > min_psnr:
                images_files = os.listdir(opt.save_folder)
                for file in images_files:
                    if file.endswith('.pth'):
                        os.remove(os.path.join(opt.save_folder, file))
                min_psnr = psnr
                model.eval()
                SC = opt.model_type + 'net_epoch_' + str(epoch) + '_'  + '.pth'
                torch.save(model.state_dict(), os.path.join(opt.save_folder, SC))
                model.train()
            lastest_model = 'ep' + str(epoch) + '.pth'
            torch.save(model.state_dict(), os.path.join(opt.save_folder, SC))