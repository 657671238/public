import os
from PIL import Image
import cv2
from torch.utils import data
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import warnings
import random
import numpy

def modcrop(image, modulo):
    h, w = image.shape[0], image.shape[1]
    h = h - h % modulo
    w = w - w % modulo
    return image[:h, :w]

def cv2_rotate(image, angle=15):
	height, width = image.shape[:2]    
	center = (width / 2, height / 2)   
	scale = 1                        
	M = cv2.getRotationMatrix2D(center, angle, scale)
	image_rotation = cv2.warpAffine(src=image, M=M, dsize=(width, height), borderValue=(0, 0, 0))
	return image_rotation



def make_augment(low_quality, high_quality):
	# 以 0.6 的概率作数据增强
	if(random.random() > 1 - 0.9):
		# 待增强操作列表(如果是 Unet 的话, 其实这里可以加入一些旋转操作)
		all_states = ['crop', 'flip', 'rotate']
		# 打乱增强的顺序
		random.shuffle(all_states)
		for cur_state in all_states:
			if(cur_state == 'flip'):
				# 0.5 概率水平翻转
				if(random.random() > 0.5):
					low_quality = cv2.flip(low_quality, 1)
					high_quality = cv2.flip(high_quality, 1)
					# print('水平翻转一次')
			elif(cur_state == 'crop'):
				# 0.5 概率做裁剪
				if(random.random() > 1 - 0.8):
					H, W, _ = low_quality.shape
					ratio = random.uniform(0.75, 0.95)
					_H = int(H * ratio)
					_W = int(W * ratio)
					pos = (numpy.random.randint(0, H - _H), numpy.random.randint(0, W - _W))
					low_quality = low_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					high_quality = high_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					# print('裁剪一次')
			elif(cur_state == 'rotate'):
				# 0.2 概率旋转
				if(random.random() > 1 - 0.1):
					angle = random.randint(-15, 15)  
					low_quality = cv2_rotate(low_quality, angle)
					high_quality = cv2_rotate(high_quality, angle)
					# print('旋转一次')
	return low_quality, high_quality

def cut_img(low,high):
    h, w = low.shape[0], low.shape[1]
    l = random.randint(0,h-224)
    r = random.randint(0,w-224)
    low_cut = low[l:l+224,r:r+224,:]
    high_cut = high[l:l+224,r:r+224,:]
    return low_cut,high_cut


class datset(Dataset):
    """NYUDataset."""

    def __init__(self, root_dir, size=14):
        """
        Args:
            root_dir (string): Directory with all the images.
            scale (float): dataset scale
            train (bool): train or test
            transform (callable, optional): Optional transform to be applied on a sample.
            数据必须被8整除

        """
        self.size = size
        self.start = 0
        self.root_dir = root_dir
        self.images = self.root_dir + '/rgb'
        self.noises = self.root_dir + '/noise'
        #self.noises = self.root_dir + '/3dLUT'
        # if train == False:
        #     self.noises = self.root_dir + '/3dLUT'
        self.transform = transforms.Compose([
           transforms.ToTensor()
        ])
        list1 = os.listdir(self.images)
        list2 = os.listdir(self.noises)
        self.lens = len(list2)//10
        list1.sort(key=lambda x: int(x[:-4]))
        list2.sort(key=lambda x: int(x[:-4]))
        self.images = [os.path.join(self.images, img) for img in list1]
        self.noises = [os.path.join(self.noises, img) for img in list2]

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
        image = self.images[idx*10 + self.start]
        noise = self.noises[idx*10 + self.start]



        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = cv2.imread(image)
            noise = cv2.imread(noise)

        image = np.asarray(image)
        noise = np.asarray(noise)
        #print(image.shape,noise.shape,type(image))
        image = modcrop(image,28) 
        noise = modcrop(noise,28) 
        if self.transform:
            image = self.transform(image).float()
            noise = self.transform(noise).float()
        #print(image.shape,noise.shape)

        sample = {'gt': image, 'target': noise}

        return sample


if __name__ == "__main__":
    test_dataset = datset('/home/dongxuan/users/xukang/SIDD_Medium_Srgb')
    
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    for idx, i in enumerate(dataloader_test):
        gt = i['gt']
        targ = i['target']
        cv2.imwrite('./images_gt/{}.png'.format(idx), gt[0].cpu().permute(1, 2, 0).numpy() * 255)
        cv2.imwrite('./images_noise/{}.png'.format(idx), targ[0].cpu().permute(1, 2, 0).numpy() * 255)


    