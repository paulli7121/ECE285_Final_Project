import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 padding, 
                 norm_layer, 
                 dropout, 
                 bias,
                 non_lin):
        super(ResidualBlock,self).__init__()
###################################################################


#         in_channels:                    # of input channels
#         out_channels:                   # of output channels
#         padding:                         padding mode
#          norm_layer, :                "batchnorm" or "instancenorm"
#         droupout:                     boolean variable, if Ture use a droup out of 0.5
#         bias:                         boolean variable, if True, set bias of conved to true
#         non_lin                       non_linearity layer


######################################################################
        super(ResidualBlock, self).__init__()
        self.block = []
        conv1 = ConvLayer(in_channels, 
                          out_channels, 
                          kernel_size=3, 
                          stride=1,
                          padding=padding,bias=bias)
        self.block += [conv1]
        
        if norm_layer == "InstanceNorm":
            normal_layer = nn.InstanceNorm2d(in_channels)
        else:
            normal_layer = nn.BatchNorm2d(in_channels)
            
        self.block += [normal_layer,non_lin]
        if dropout:
            self.block += [nn.Dropout(0.5)]
        self.block += [conv1]
        self.block += [normal_layer]
        self.forward_func = nn.Sequential(*self.block)
    
    def forward(self, x):
        return x + self.forward_func(x)
    
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, 
                 out_channels,  
                 kernel_size,
                 stride,
                 padding,
                 bias):
        super(ConvLayer, self).__init__()
        self.block = []
        p = 0
        if padding == "reflect":
            self.block.append(nn.ReflectionPad2d(kernel_size//2))

        elif padding == "replicate":
            self.block.append(nn.ReplicationPad2d(kernel_size//2))

        else:
            p = 1
        self.block.append( nn.Conv2d(in_channels,
                                    out_channels, 
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=p, 
                                    bias=bias) )
        self.forward_func = nn.Sequential(*self.block)

    def forward(self,x):
        return self.forward_func(x)