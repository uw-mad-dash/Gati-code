import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch
import os
from threading import Thread

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, blockIndex="-1", dataDump=False, dumpPath="emptyPath", num_classes = 10,use_freeze_inference= 0, dumpLayer=0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_freeze_inference = use_freeze_inference
        self.num_classes = num_classes
        self.dumpData = dataDump
        self.blockNum = blockIndex
        self.dumpLayer = dumpLayer
        if self.dumpData:
            self.file = open(os.path.join(dumpPath, "LayerOutput_block_{}".format(self.blockNum)), "w")


    def forward(self, x):
        if len(x[0]) == self.num_classes:
            return x
        residual = x
        batchSize = x.shape[0]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        if self.dumpData and self.blockNum == self.dumpLayer:
            np.savetxt(self.file, out.view(batchSize, -1).cpu().detach().numpy(), fmt='%1.9f', delimiter=',')
        if self.use_freeze_inference == 1 and self.blockNum > 2 and self.blockNum < 6:
            intermediate = out.view(batchSize, -1)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            input = intermediate.to(device)
            DRinput = out.float()
            output = self.DRmodels[self.blockNum](input)
            _, predicted = output.max(1)
            np_prob = output.cpu().numpy()

            if np.amax(np_prob[0]) > self.thresholds[self.blockNum-3][predicted.item()]:
                print("the point is frozen:", self.blockNum)
                self.frozenBlock = self.blockNum
                
                return output
            
            else:
                return out
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, blockIndex="-1", dataDump=False, dumpPath="emptyPath", num_classes = 10,use_freeze_inference= 0, dumpLayer=0):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.use_freeze_inference = use_freeze_inference
        self.num_classes = num_classes
        self.dumpData = dataDump
        self.blockNum = blockIndex
        self.dumpLayer = dumpLayer
        if self.dumpData:
            self.file = open(os.path.join(dumpPath, "LayerOutput_block_{}".format(self.blockNum)), "w")


    def forward(self, x):
        if len(x[0]) == self.num_classes:
            return x
        residual = x
        batchSize = x.shape[0]

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        if self.dumpData and self.blockNum == self.dumpLayer:
            np.savetxt(self.file, out.view(batchSize, -1).cpu().detach().numpy(), fmt='%1.9f', delimiter=',')
        if self.use_freeze_inference == 1 and self.blockNum > 2 and self.blockNum < 6:
            intermediate = out.view(batchSize, -1)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            input = intermediate.to(device)
            DRinput = out.float()
            output = self.DRmodels[self.blockNum](input)
            _, predicted = output.max(1)
            np_prob = output.cpu().numpy()

            if np.amax(np_prob[0]) > self.thresholds[self.blockNum-3][predicted.item()]:
                print("the point is frozen:", self.blockNum)
                self.frozenBlock = self.blockNum
                
                return output
            
            else:
                return out
        else:
            return out

        


class ResNet(nn.Module):

    def __init__(self, block, layers, dumpData=False, dumpPath="emptyPath", num_classes=10, use_freeze_inference= 0, dumpLayer=0):
        super(ResNet, self).__init__()
        
        # if dumpLayer
        self.dumpData = dumpData
        self.dumpPath  = dumpPath
        
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64,  layers[0], dumpData=self.dumpData, dumpPath=self.dumpPath, blockNum=0,use_freeze_inference= use_freeze_inference, dumpLayer=dumpLayer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dumpData=self.dumpData, dumpPath=self.dumpPath, blockNum=layers[0],use_freeze_inference= use_freeze_inference, dumpLayer=dumpLayer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dumpData=self.dumpData, dumpPath=self.dumpPath, blockNum=layers[0]+layers[1],use_freeze_inference= use_freeze_inference, dumpLayer=dumpLayer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dumpData=self.dumpData, dumpPath=self.dumpPath, blockNum=layers[0]+layers[1]+layers[2],use_freeze_inference= use_freeze_inference, dumpLayer=dumpLayer)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.num_classes = 100
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

        self.checkpoints = {
                                "block_1":False,
                                "block_2":False,
                                "block_3":False,
                                "block_4":False,
                           }
        
        self.layerDumps = {}
        self.acc = 0
        

    def addLayerDumps(self, block_name, y):
            if self.checkpoints[block_name]==True:
                self.layerDumps[block_name] = y


    def _make_layer(self, block, planes, blocks, stride=1, dumpData=False, dumpPath="emptyPath", blockNum=-1, num_classes = 10, use_freeze_inference= 0, dumpLayer=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, blockIndex=blockNum, dataDump=dumpData, dumpPath=dumpPath, num_classes = num_classes, dumpLayer=dumpLayer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            blockNum += 1 
            layers.append(block(self.inplanes, planes, blockIndex=blockNum, dataDump=dumpData, dumpPath=dumpPath, num_classes = num_classes, dumpLayer=dumpLayer))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
       
        
        if len(x[0]) == self.num_classes:
            return x,1
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x,0


def resnet18(pretrained=False, dumpData=False, dumpPath="emptyPath",use_freeze_inference= 0,dumpLayer=0, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    pretrained = False
    model = ResNet(BasicBlock, [2, 2, 2, 2], dumpData, dumpPath,use_freeze_inference= use_freeze_inference, dumpLayer=dumpLayer, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, dumpData=False, dumpPath="emptyPath",**kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    pretrained = False
    model = ResNet(BasicBlock, [3, 4, 6, 3], dumpData, dumpPath, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, dumpData=False, dumpPath="emptyPath",use_freeze_inference= 0,dumpLayer=0, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    pretrained = False
    model = ResNet(Bottleneck, [3, 4, 6, 3], dumpData, dumpPath,use_freeze_inference= use_freeze_inference, dumpLayer=dumpLayer, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, dumpData=False, dumpPath="emptyPath", **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], dumpData, dumpPath, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, dumpData=False, dumpPath="emptyPath",use_freeze_inference= 0,dumpLayer=0, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], dumpData, dumpPath,use_freeze_inference= use_freeze_inference, dumpLayer=dumpLayer, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
