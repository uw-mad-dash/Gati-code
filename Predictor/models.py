import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
class ClassifierLayer(nn.Module):
    def __init__(self, inFeatures, outFeatures, hiddenSize=1024):
        super(ClassifierLayer, self).__init__()
        self.layer1 = nn.Linear(inFeatures, hiddenSize)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(hiddenSize, outFeatures)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = F.log_softmax(x)
        return x


                
        
class Classifier3(nn.Module):
    def __init__(self, inFeatures, outFeatures):
        super(Classifier3, self).__init__()
        self.layer1 = nn.Linear(inFeatures, 1024)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(1024, outFeatures)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = F.softmax(x)
        return x    

class ConvClassifier(nn.Module):
    def __init__(self, inFeatures, outFeatures, kernel_size=5, stride=5, padding = 0, dilation=1):
        super(ConvClassifier, self).__init__()
        self.layer1 = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()
        if (inFeatures + 2*padding - dilation * (kernel_size - 1)) % stride == 0:
            self.hidden_size = int((inFeatures + 2*padding - dilation * (kernel_size - 1))/stride)
        else:
            self.hidden_size = math.floor((inFeatures + 2*padding - dilation * (kernel_size - 1)) / stride) + 1
        self.layer2 = nn.Linear(self.hidden_size, outFeatures)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.layer2(x)
        x = F.log_softmax(x)
        return x
    
class PoolingClassifier(nn.Module):
    def __init__(self, inFeatures, outFeatures, afterPoolingSize, hiddenLayerSize=1024):
        super(PoolingClassifier, self).__init__()

        self.layer1 = nn.Sequential(
            nn.AdaptiveAvgPool1d(afterPoolingSize),
            nn.Linear(afterPoolingSize,hiddenLayerSize)
        )
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hiddenLayerSize, outFeatures)

    def forward(self, x):
        x = self.layer1(x)
        
        x = self.relu(x)
        x = x.view(x.shape[0], -1)

        x = self.layer2(x)
        x = F.log_softmax(x)
        return x
