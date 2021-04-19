import os
import argparse 
import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import utils
import torch, torchvision

import torchvision.transforms as transforms


# this is not creating numpy array
# also all the parsing and splitting are happening in __get_item__
def loadData(filename):
    data = []
    count = 0

    with open(filename) as f:
        print("Goint to read file")
       
        for idx, line in enumerate(f):
            count = count + 1
            if count%10000==0:
                print(count)
            data = data + [line]
#            if count%5000==0:
#               return data
    return data

def readline(filename, idx):
    with open(filename) as f:
        print(idx)
        for i, line in enumerate(f):
            if i == idx:
#                 print(len([float(x) for x in line.strip().split(",")]))
                return [float(x) for x in line.strip().split(",")]


class DataNoNumpy(data.Dataset):
    def __init__(self, trainData, trainLabels):
        super(DataNoNumpy, self).__init__()
        print("Started in Data")
        self.data = loadData(trainData)
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


def GetTrainTestData(dataset, batch_size, dataset_dir, shuffle_train = True):
    
    # define transform for train data
    # mean and std calculated from utils/GetMeanNdStd
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        # load datasets
        trainData = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform_train)
        # sampler = list(range(64*10))
        trainDataGen = DataLoader(trainData, batch_size=batch_size, num_workers=1, shuffle=shuffle_train)#, sampler=Sampler.SubsetRandomSampler(sampler))


        # change transform for test dataset
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testData = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform_test)
        testDataGen = DataLoader(testData, batch_size=batch_size, num_workers=1, shuffle=False)
        
        return trainDataGen, testDataGen 

    elif dataset == "cifar100":
        transform = transforms.Compose(
                                    [
                                    # transforms.Resize(224),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        # load datasets
        trainData = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True, transform=transform)
        # sampler = list(range(64*10))
        trainDataGen = DataLoader(trainData, batch_size=batch_size, num_workers=1, shuffle=False)#, sampler=Sampler.SubsetRandomSampler(sampler))

        # change transform for test dataset
        transform = transforms.Compose([
                                        # transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        testData = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=transform)
        testDataGen = DataLoader(testData, batch_size=batch_size, num_workers=1, shuffle=False)

        return trainDataGen, testDataGen
