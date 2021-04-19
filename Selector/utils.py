import os
import argparse 
import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import model
import pandas as pd
import utils

def loadData(filename, make_numpy=True):
    data = []
    count = 0
    
    with open(filename) as f:
        print("Goint to read file")
        lines = f.readlines()
        print("aad")
        for line in lines:
            count = count + 1
            if count%100==0:
                print(count)  
            
            row = [float(x) for x in line.strip().split(",")]
            data = data + [row]
#            if count%45000==0:
#		return np.array(data)
    if make_numpy:
        return np.array(data)    
    else:
        return data

def loadData_II(filename):
    data = []
    count = 0

    with open(filename) as f:
        print("Goint to read file")
       
        for idx, line in enumerate(f):
            count = count + 1
            if count%100==0:
                print(count)
            data = data + [line]
#            if count%5000==0:
#               return data
    return data

class Data(data.Dataset):
    def __init__(self, trainData, trainLabels):
        super(Data, self).__init__()
        print("Started in Data")
        self.data = loadData(trainData, make_numpy=True)
        print("data loaded")
        self.labels = np.genfromtxt(trainLabels, delimiter=',')
        print("Done")
    
    def __getitem__(self, idx):
        return self.data[idx,], self.labels[idx]
    
    def __len__(self):
        return self.data.shape[0]
    
    def inputDim(self):
        return self.data.shape[1]

class DataNoNumpy(data.Dataset):
    def __init__(self, trainData, trainLabels):
        super(DataNoNumpy, self).__init__()
        print("Started in Data")
        self.data = loadData_II(trainData)
        print("data loaded")
        self.labels = np.genfromtxt(trainLabels, delimiter=',')
        print("Done")

    def __getitem__(self, idx):
        line = [float(x) for x in self.data[idx].split(',')]
        return np.array(line), self.labels[idx]

    def __len__(self):
        return len(self.data)

    def inputDim(self):
        if len(self.data)!=0:
            line = [x for x in self.data[0].split(',')]
            return len(line)
        else:
            print("DataLoader:Data EmptY!")
            return -1
    def getLabels(self):
        return self.labels
        
def generateSelectorLabels(labels, realLabels):
    """ 
    args: 
        labels: the generated labels of the DR models (the class associated with the max score)
        predictedLabelsFileName: the real predicted labels 
    """
    selector_labels = []
    for i in range(len(labels)):
        if labels[i] == realLabels[i]:
            selector_labels.append(1)
        else:
            selector_labels.append(0)
    print(selector_labels)
    return np.array(selector_labels)

class DrData(data.Dataset):
    def __init__(self, data, labels, realLabels):
        super(DrData, self).__init__()
        print("Started to create data samples.")
        self.data = data
        print("data loaded")
        self.labels = generateSelectorLabels(labels, realLabels)
    
    def __getitem__(self, idx):
        return self.data[idx,], self.labels[idx]
    
    def __len__(self):
        return self.data.shape[0]
    
    def inputDim(self):
        return self.data.shape[1]