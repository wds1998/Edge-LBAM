import torch
from PIL import Image
from os import listdir, walk
from os.path import join
from random import randint
from data.basicFunction import CheckImageFile, ImageTransform, MaskTransform
import numpy as np
import torchvision.transforms.functional as F
import random
from skimage.feature import canny
from skimage.color import rgb2gray
from shutil import copyfile
from scipy.misc import imread
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class GetData(Dataset):
    def __init__(self, dataRoot, maskRoot, loadSize, cropSize):
        super(GetData, self).__init__()

        self.imageFiles = [join(dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
                           for files in filenames if CheckImageFile(files)]
        self.masks = [join(dataRootK, files) for dataRootK, dn, filenames in walk(maskRoot) \
                      for files in filenames if CheckImageFile(files)]
        self.numOfMasks = len(self.masks)
        self.loadSize = loadSize
        self.cropSize = cropSize
        self.ImgTrans = ImageTransform(loadSize, cropSize)
        self.maskTrans = MaskTransform(cropSize)
        self.sigma = 1.5

    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        randnum = randint(0, self.numOfMasks - 1)
        # mask = Image.open(self.imageFiles[index].replace("GT","mask"))
        mask = Image.open(self.masks[randnum])
        groundTruth = self.ImgTrans(img.convert('RGB'))
        mask = self.maskTrans(mask.convert('RGB'))
        # we add this threshhold to force the input mask to be binary 0,1 values
        # the threshhold value can be changeble, i think 0.5 is ok
        threshhold = 0.5
        ones = mask >= threshhold
        zeros = mask < threshhold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        # here, we suggest that the white values(ones) denotes the area to be inpainted,
        # and dark values(zeros) is the values remained.
        # Therefore, we do a reverse step let mask = 1 - mask, the input = groundTruth * mask, :).
        edge_mask = np.transpose(mask, (1, 2, 0))
        mask = 1 - mask
        inputImage = groundTruth * mask
        edge_mask = edge_mask.numpy()
    
        edge_mask = rgb2gray(edge_mask)
        edge_mask2 = (edge_mask > 0).astype(np.uint8) * 255  # threshold due to interpolation

        tmp = np.transpose(groundTruth, (1, 2, 0))
        tmp = tmp.numpy()
        img_gray = rgb2gray(tmp)

        edge = self.load_edge(img_gray, np.array(1 - edge_mask2 / 255).astype(np.bool))
        img_gray = torch.from_numpy(img_gray.reshape((1, 256, 256)))
        edge_mask = torch.from_numpy((edge_mask).reshape((1,256,256)))

        edge = torch.from_numpy(edge.reshape((1, 256, 256))).float()
        inputImage = torch.cat((inputImage, mask[0].view(1, 256, 256)), 0)

        return inputImage,groundTruth, mask, img_gray, edge, edge_mask.float(),self.imageFiles[index]

    def __len__(self):
        return len(self.imageFiles)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def load_edge(self, img, mask):
        sigma = self.sigma
        # in test mode images are masked (with masked regions),
        # using 'mask' parameter prevents canny to detect edges for the masked regions

        # canny

        # no edge
        if sigma == -1:
            return np.zeros(img.shape).astype(np.float)

        # random sigma
        if sigma == 0:
            sigma = random.randint(1, 4)

        return canny(img, sigma=sigma, mask=mask).astype(np.float)




