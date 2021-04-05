import torch
import torch.nn as nn
from torchvision import models
from models.forwardAttentionLayer import ForwardAttention
from models.reverseAttentionLayer import ReverseAttention, ReverseMaskConv
from models.weightInitial import weights_init
from models.EdgeAttentionLayer import ForwardEdgeAttention
from models.weightInitial import weights_init
#VGG16 feature extract
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        # vgg16.load_state_dict(torch.load('../vgg16-397923af.pth'))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class LBAMModel(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        super(LBAMModel, self).__init__()
        self.maskconv1 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2,False)
        )
        self.maskconv2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2,False)
        )
        self.edgeconv1 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2,False)
        )
        self.edgeconv2 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2,False)
        )
        self.maskconv3 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2,False)
        )
        self.maskconv4 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2,False)
        )
        self.edgeconv3 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2,False)
        )
        self.edgeconv4 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.2,False)
        )
        self.maskconv1.apply(weights_init())
        self.maskconv2.apply(weights_init())
        self.maskconv3.apply(weights_init())
        self.maskconv4.apply(weights_init())
        self.edgeconv1.apply(weights_init())
        self.edgeconv2.apply(weights_init())
        self.edgeconv3.apply(weights_init())

        # default kernel is of size 4X4, stride 2, padding 1, 
        # and the use of biases are set false in default ReverseAttention class.
        self.ec1 = ForwardAttention(5, 64, bn=False)
        self.ec2 = ForwardAttention(64, 128)
        self.ec3 = ForwardAttention(128, 256)
        self.ec4 = ForwardAttention(256, 512)
        self.edge1 = ForwardEdgeAttention(3,64)
        self.edge2 = ForwardEdgeAttention(64,128)
        self.edge3 = ForwardEdgeAttention(128,256)
        self.edge4 = ForwardEdgeAttention(256,512)
        for i in range(5, 8):
            name = 'ec{:d}'.format(i)
            setattr(self, name, ForwardAttention(512, 512))
            name2 = 'edge{:d}'.format(i)
            setattr(self,name2,ForwardEdgeAttention(512,512))
        
        # reverse mask conv
        self.reverseConv1 = ReverseMaskConv(3, 64)
        self.reverseConv2 = ReverseMaskConv(64, 128)
        self.reverseConv3 = ReverseMaskConv(128, 256)
        self.reverseConv4 = ReverseMaskConv(256, 512)
        self.reverseConv5 = ReverseMaskConv(512, 512)
        self.reverseConv6 = ReverseMaskConv(512, 512)
        self.reverseedge1 = ForwardEdgeAttention(3,64)
        self.reverseedge2 = ForwardEdgeAttention(64,128)
        self.reverseedge3 = ForwardEdgeAttention(128,256)
        self.reverseedge4 = ForwardEdgeAttention(256,512)
        self.reverseedge5 = ForwardEdgeAttention(512, 512)
        self.reverseedge6 = ForwardEdgeAttention(512, 512)
        self.dc1 = ReverseAttention(512, 512, bnChannels=1024)
        self.dc2 = ReverseAttention(512 * 2, 512, bnChannels=1024)
        self.dc3 = ReverseAttention(512 * 2, 512, bnChannels=1024)
        self.dc4 = ReverseAttention(512 * 2, 256, bnChannels=512)
        self.dc5 = ReverseAttention(256 * 2, 128, bnChannels=256)
        self.dc6 = ReverseAttention(128 * 2, 64, bnChannels=128)
        self.dc7 = nn.ConvTranspose2d(64 * 2, outputChannels, kernel_size=4, stride=2, padding=1, bias=False)

        self.tanh = nn.Tanh()

    def forward(self, inputImgs, masks,edge):
        mask1 = self.maskconv1(masks)
        mask1 =self.maskconv2(mask1)
        edge1 = self.edgeconv1(edge)
        edge1 = self.edgeconv2(edge1)
        maskoutput1,edgeoutput,feature1 = self.edge1(mask1,edge1)
        ef, mu1, skipConnect1, forwardMap1 = self.ec1(inputImgs, maskoutput1)
        maskoutput,edgeoutput,feature2 = self.edge2(mu1,edgeoutput)
        ef, mu2, skipConnect2, forwardMap2 = self.ec2(ef, maskoutput)
        maskoutput3,edgeoutput,feature3 = self.edge3(mu2,edgeoutput)
        ef, mu3, skipConnect3, forwardMap3 = self.ec3(ef, maskoutput3)
        maskoutput,edgeoutput,_ = self.edge4(mu3,edgeoutput)
        ef, mu, skipConnect4, forwardMap4 = self.ec4(ef, maskoutput)
        maskoutput,edgeoutput,_ = self.edge5(mu,edgeoutput)
        ef, mu, skipConnect5, forwardMap5 = self.ec5(ef, maskoutput)
        maskoutput,edgeoutput,_ = self.edge6(mu,edgeoutput)
        ef, mu, skipConnect6, forwardMap6 = self.ec6(ef, maskoutput)
        maskoutput, edgeoutput,_ = self.edge7(mu, edgeoutput)
        ef, _, _, _ = self.ec7(ef, maskoutput)

        mask2 = self.maskconv3(1-masks)
        mask2 = self.maskconv4(mask2)
        edge2 = self.edgeconv3(edge)
        edge2 = self.edgeconv4(edge2)
        maskoutput1,edgeoutput,feature1 = self.reverseedge1(mask2,edge2)
        reverseMap1, revMu = self.reverseConv1(maskoutput1)
        maskoutput2,edgeoutput,feature2 = self.reverseedge2(revMu,edgeoutput)
        reverseMap2, revMu = self.reverseConv2(maskoutput2)
        maskoutput3, edgeoutput,feature3 = self.reverseedge3(revMu, edgeoutput)
        reverseMap3, revMu = self.reverseConv3(maskoutput3)
        maskoutput, edgeoutput,_ = self.reverseedge4(revMu, edgeoutput)
        reverseMap4, revMu = self.reverseConv4(maskoutput)
        maskoutput, edgeoutput,_ = self.reverseedge5(revMu, edgeoutput)
        reverseMap5, revMu = self.reverseConv5(maskoutput)
        maskoutput, edgeoutput,_ = self.reverseedge6(revMu, edgeoutput)
        reverseMap6, _ = self.reverseConv6(maskoutput)

        concatMap6 = torch.cat((forwardMap6, reverseMap6), 1)
        dcFeatures1 = self.dc1(skipConnect6, ef, concatMap6)

        concatMap5 = torch.cat((forwardMap5, reverseMap5), 1)
        dcFeatures2 = self.dc2(skipConnect5, dcFeatures1, concatMap5)

        concatMap4 = torch.cat((forwardMap4, reverseMap4), 1)
        dcFeatures3 = self.dc3(skipConnect4, dcFeatures2, concatMap4)

        concatMap3 = torch.cat((forwardMap3, reverseMap3), 1)
        dcFeatures4 = self.dc4(skipConnect3, dcFeatures3, concatMap3)

        concatMap2 = torch.cat((forwardMap2, reverseMap2), 1)
        dcFeatures5 = self.dc5(skipConnect2, dcFeatures4, concatMap2)

        concatMap1 = torch.cat((forwardMap1, reverseMap1), 1)
        dcFeatures6 = self.dc6(skipConnect1, dcFeatures5, concatMap1)

        dcFeatures7 = self.dc7(dcFeatures6)

        output = torch.abs(self.tanh(dcFeatures7))

        return output,forwardMap1,forwardMap2,forwardMap3, reverseMap1,reverseMap2,reverseMap3