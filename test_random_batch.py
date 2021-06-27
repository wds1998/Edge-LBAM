import os
import math
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data.dataloader import GetData
from models.LBAMModel import LBAMModel
import pytorch_ssim
import random
import numpy as np
from MECNet.models import EdgeModel
from MECNet.config import Config
import numpy
from PIL.Image import fromarray

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
numpy.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=4,
                    help='workers for dataloader')
parser.add_argument('--local_rank',type=int,default=0)
parser.add_argument('--pretrained', type=str, default='', help='pretrained models')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=350,
                    help='image loading size')
parser.add_argument('--cropSize', type=int, default=256,
                    help='image training size')
parser.add_argument('--dataRoot', type=str,
                    default='')
parser.add_argument('--maskRoot', type=str,
                    default='')
parser.add_argument('--savePath', type=str, default='./results')
args = parser.parse_args()

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda is available!')
    cudnn.benchmark = True
os.makedirs(os.path.join(args.savePath,"GT"), exist_ok=True)
os.makedirs(os.path.join(args.savePath,"damaged"), exist_ok=True)
os.makedirs(os.path.join(args.savePath,"ours"), exist_ok=True)
os.makedirs(os.path.join(args.savePath,"input"), exist_ok=True)
os.makedirs(os.path.join(args.savePath,"masks"), exist_ok=True)
os.makedirs(os.path.join(args.savePath,"edge"), exist_ok=True)


batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)
cropSize = (args.cropSize, args.cropSize)
dataRoot = args.dataRoot
maskRoot = args.maskRoot
savePath = args.savePath

if not os.path.exists(savePath):
    os.makedirs(savePath)

config = Config("config.yml")
edge_model = EdgeModel(config).to(config.DEVICE)
edge_model.load()
edge_model.cuda()
edge_model = nn.DataParallel(edge_model, device_ids=[0,1])
imgData = GetData(dataRoot, maskRoot, loadSize, cropSize)
data_loader = DataLoader(imgData, batch_size=batchSize, shuffle=False, num_workers=1, drop_last=False)

num_epochs = 100

netG = LBAMModel(5, 3)

if args.pretrained != '':
    netG.load_state_dict(torch.load(args.pretrained))
else:
    print('No pretrained model provided!')

#
if cuda:
    netG = netG.cuda()

for param in netG.parameters():
    param.requires_grad = False

print('OK!')


sum_psnr = 0
sum_ssim = 0
count = 0
sum_time = 0.0
l1_loss = 0

import time
start = time.time()
for i in range(1, num_epochs + 1):
    netG.eval()
    for inputImgs, GT, masks, img_gray,edge,masks_over in (data_loader):
        if count >= 60:
            break
        if cuda:
            inputImgs = inputImgs.cuda()
            img_gray=img_gray.cuda()
            GT = GT.cuda()
            masks = masks.cuda()
            edge = edge.cuda()
            masks_over=masks_over.cuda()
        outputs_2 = edge_model(img_gray, edge, masks_over)
        outputs_merged = (outputs_2 * masks_over) + (edge * (1 - masks_over))
        inputImgs2 = torch.cat((inputImgs, outputs_merged), 1)
        #do something other
        fake_images = netG(inputImgs2, masks,outputs_merged)

        g_image = fake_images.data.cpu()
        GT = GT.data.cpu()
        mask = masks.data.cpu()
        damaged = GT * mask
        generaredImage = GT * mask + g_image * (1 - mask)
        groundTruth = GT
        masksT = mask
        generaredImage = generaredImage
        groundTruth = groundTruth
        count += 1
        batch_mse = ((groundTruth - generaredImage) ** 2).mean()
        psnr = 10 * math.log10(1 / batch_mse)
        sum_psnr += psnr
        print(count, ' psnr:', psnr)
        ssim = pytorch_ssim.ssim(groundTruth * 255, generaredImage * 255)
        sum_ssim += ssim
        print(count, ' ssim:', ssim)
        l1_loss += nn.L1Loss()(generaredImage, groundTruth)

        outputs =torch.Tensor(5* GT.size()[0], GT.size()[1], cropSize[0], cropSize[1])
        for i in range(GT.size()[0]):
            outputs[5 * i] = masksT[i]
            outputs[5 * i + 1] = damaged[i]
            outputs[5 * i + 2] = GT[i] * masksT[i]
            outputs[5 * i + 2] = generaredImage[i]
            outputs[5 * i + 3] = GT[i]
            outputs[5 * i + 4]=outputs_merged[i]
            #outputs[5 * i + 4] = 1 - masksT[i]
        # save_image(outputs,  os.path.join(savePath, 'results-{}'.format(count) + '.png'))

        # make subdirs to save mask GT results and input and damaged images
        damaged = GT * mask + (1 -  mask)

        for j in range(GT.size()[0]):
            save_image(outputs[5 * j + 1], savePath + '/damaged/damaged{}-{}.png'.format(count, j))
            outputs[5 * j + 1] = damaged[j]

        for j in range(GT.size()[0]):
            outputs[5 * j] = 1- masksT[j]
            save_image(outputs[5 * j], savePath + '/masks/mask{}-{}.png'.format(count, j))
            save_image(outputs[5 * j + 1], savePath + '/input/input{}-{}.png'.format(count, j))
            save_image(outputs[5 * j + 2], savePath + '/ours/ours{}-{}.png'.format(count, j))
            save_image(outputs[5 * j + 3], savePath + '/GT/GT{}-{}.png'.format(count, j))
            save_image(outputs[5 * j + 4], savePath + '/edge/edge{}-{}.png'.format(count, j))



end = time.time()
sum_time += (end - start) / batchSize


print('avg l1 loss:', l1_loss / count)
print('average psnr:', sum_psnr / count)
print('average ssim:', sum_ssim / count)
print('average time cost:', sum_time / count)
