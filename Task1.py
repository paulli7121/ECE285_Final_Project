
get_ipython().run_line_magic('matplotlib', 'notebook')
import os
import numpy as np
from glob import glob
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
import nntools as nt
from PIL import Image
from matplotlib import pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms


class styleLoader():
    def __init__(self, root_dir, image_size=(512, 512)):
        self.image_size = image_size
        self.images_dir = root_dir
    def __getitem__(self, idx):
        img = Image.open(self.images_dir)
        transform = tv.transforms.Compose([
            # COMPLETE
            tv.transforms.Resize(self.image_size),
            tv.transforms.ToTensor(),
            ])
        x = transform(img)
        x = x.unsqueeze(0)
        return x.to(device, torch.float)

def myimshow(image, ax=plt):
    image = image.to('cpu').clone().numpy()
    image = image.squeeze(0)
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    h = ax.imshow(image)
    ax.axis('off')
    return h

class contentLoss(nn.Module):
    def __init__(self, target):
        super(contentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        N = input.shape[-1]
        self.loss = F.mse_loss(input, self.target)*N*N*0.5
        return input

def gram(input):
    batch_size, channel, height, width = input.shape
    x = input.view(batch_size*channel, height*width)
    return torch.mm(x, x.t())/(batch_size*channel)

class styleLoss(nn.Module):
    def __init__(self, target):
        super(styleLoss, self).__init__()
        self.target = target
        
    def forward(self, input):
        target_gram = gram(self.target)
        input_gram = gram(input)
        self.loss = F.mse_loss(input_gram, target_gram)/4
        return input
    

def constructModel(model, contentImg, styleImg, contentLayers, styleLayers):
    contentLoss_list = []
    styleLoss_list = []

    newModel = nn.Sequential()

    convCount = 0
    poolCount = 0
    reluCount = 0
    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            convCount += 1
            name = 'conv_'+str(convCount)
            newModel.add_module(name, layer)
            if name in contentLayers:
                content = newModel(contentImg)
                content_loss = contentLoss(content)
                newModel.add_module("content_loss_"+str(convCount), content_loss)
                contentLoss_list.append(content_loss)

            if name in styleLayers:
                style = newModel(styleImg).detach()
                style_loss = styleLoss(style)
                newModel.add_module("style_loss_"+str(convCount), style_loss)
                styleLoss_list.append(style_loss)
        if isinstance(layer, nn.ReLU):
            reluCount += 1
            name = 'relu_'+str(reluCount)
            layer = nn.ReLU(inplace=False)
            newModel.add_module(name, layer)
        if isinstance(layer, nn.MaxPool2d):
            poolCount += 1
            name = 'pool_'+str(poolCount)
            newModel.add_module(name, nn.AvgPool2d((2,2)))

    return newModel, styleLoss_list, contentLoss_list


def trainHelper(contentImg, styleImg, epoch, model, contentLayers, styleLayers, beta):
    trainImg = contentImg
    input_param = trainImg.requires_grad_()
    optimizer = optim.LBFGS([input_param])
    newModel,styleLoss_list,contentLoss_list = constructModel(model, contentImg, styleImg, contentLayers, styleLayers )
    i = [0]
    loss1 = []
    loss2 = []
    while i[0] <= epoch:
        def closure():
            trainImg.data.clamp(0,1)
            optimizer.zero_grad()
            newModel(trainImg)
            content_loss = 0
            style_loss = 0
            for loss_val in contentLoss_list:
                content_loss += loss_val.loss
            for loss_val in styleLoss_list:
                style_loss += loss_val.loss
            totalLoss = content_loss * 1 + style_loss *beta
            totalLoss.backward()
            i[0] += 1
            loss1.append(content_loss * 1)
            loss2.append(style_loss * beta)
            return totalLoss
        optimizer.step(closure)
    trainImg.data.clamp_(0,1)
    return trainImg, loss1, loss2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vgg = tv.models.vgg19(pretrained=True).features.to(device).eval()

contentImg = styleLoader("house.jpg")[0]
styleImg = styleLoader("starry_night.jpg")[0]
plt.figure()
myimshow(contentImg, ax=plt)
plt.figure()
myimshow(styleImg, ax=plt)

contentLayers = ['conv_4']
styleLayers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
outputImg, loss1, loss2 = trainHelper(contentImg, styleImg, 300, vgg, contentLayers, styleLayers, 10000)
plt.figure()
myimshow(outputImg.detach(), ax=plt)

