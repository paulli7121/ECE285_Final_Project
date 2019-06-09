import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
from ResBlock import ResidualBlock
class ResNetGen(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 ngf,
                 padding, 
                 norm_layer, 
                 dropout, 
                 num_resblocks,
                 num_down_samp):
        super(ResNetGen,self).__init__()
#############################################################################################


#         in_channels:                    # of input channels
#         out_channels:                   # of output channels
#         ngf                             # of filters in the first conv layer
#         padding:                         padding mode
#         norm_layer:                      batch normal layer created outside of this class
#         droupout:                     boolean variable, if Ture use a droup out of 0.5
#         bias:                         boolean variable, if True, set bias of conved to true
#         num_resblocks                 number of residual blocks
#         num_down_samp                 number of down sampling
#                                       each contains a conv with stride 2


#############################################################################################
        bias = True
        net = [nn.ReflectionPad2d(4),
                 nn.Conv2d(in_channels,ngf, kernel_size=9,padding=0,bias=bias)
                ]
        if norm_layer == "InstanceNorm":
            net += [nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]
        else:
            net += [nn.BatchNorm2d(ngf),
                 nn.ReLU(True)]
        
        # downsampling
        for i in range(num_down_samp):
            mult = 2**i
            net += [nn.Conv2d(ngf*mult,ngf*mult*2, kernel_size=3, 
                              stride=2,padding=1, bias=bias)]
                   
            if norm_layer == "InstanceNorm":
                net += [nn.InstanceNorm2d(ngf*mult*2),
                     nn.ReLU(True)]
            else:
                net += [nn.BatchNorm2d(ngf*mult*2),
                     nn.ReLU(True)]  
                
        mult = mult * 2
        
        for i in range(num_resblocks):
            net += [ResidualBlock(in_channels=ngf*mult, 
                 out_channels=ngf*mult, 
                 padding=padding, 
                 norm_layer=norm_layer, 
                 dropout=dropout, 
                 bias=bias,
                 non_lin=nn.ReLU(True))]
            
        # upsampling   
        for i in range(num_down_samp):
            mult = 2**(num_down_samp-i)
            net += [nn.ConvTranspose2d(ngf*mult, 
                                       ngf*mult//2,
                                      kernel_size = 3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1,
                                      bias = bias)]
                   
            if norm_layer == "InstanceNorm":
                net += [nn.InstanceNorm2d(ngf*mult//2),
                     nn.ReLU(True)]
            else:
                net += [nn.BatchNorm2d(ngf*mult//2),
                     nn.ReLU(True)]  

            
        net += [nn.ReflectionPad2d(4),
               nn.Conv2d(ngf,out_channels,kernel_size=9,padding=0),
               nn.Tanh()]
        self.net = nn.Sequential(*net)
        
    def forward(self,x):
        return self.net(x)