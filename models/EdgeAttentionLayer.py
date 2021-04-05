import torch
from models.weightInitial import weights_init
from torch import nn
import torch.nn.functional as F
class ForwardEdgeAttention(nn.Module):
    def __init__(self, channels,outchannels):
        super(ForwardEdgeAttention, self).__init__()
        self.maskconv = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2,False)
        )
        self.edgegradient = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2,False)
        )
        self.edgemaskcoincide = nn.Sequential(
            nn.Conv2d(channels*2,1,kernel_size=1,padding=0,stride=1,bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2,False)
        )
        self.edgeconv = nn.Sequential(
            nn.Conv2d(channels,outchannels,kernel_size=3,padding=1,stride = 1,bias=False),
            nn.BatchNorm2d(outchannels),
            nn.LeakyReLU(0.2,False)
        )
        self.maskconv.apply(weights_init())
        self.edgegradient.apply(weights_init())
        self.edgemaskcoincide.apply(weights_init())
        self.edgeconv.apply(weights_init())
    def forward(self, mask, edge):
        # print(edge.shape)
        output2 = F.interpolate(edge,size=[mask.shape[2],mask.shape[2]])
        maskout = self.maskconv(mask)
        edge_gradient=self.edgegradient(output2)
        edge_mask_concat = torch.cat((edge_gradient,mask),1)
        edgeout = self.edgemaskcoincide(edge_mask_concat)
        maskmulti = maskout*edgeout
        output1 = maskmulti+mask
        output2 = self.edgeconv(output2)
        return output1,output2,maskmulti
