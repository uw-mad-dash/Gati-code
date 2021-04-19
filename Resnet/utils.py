import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import sys
import math

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from scipy.io import loadmat
from Resnet import resnet18, resnet152, resnet50
root_directory = "/data/ILSVRC2011"
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def GetMeanNdStd(dataset):
    mean = []
    std = []

    first_chanl = []
    second_chanl = []
    third_chanl = []


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    first = True
    count = 0
    for input, targets in dataloader:
        if first:
            first = False
            data = input
        else:
            data = torch.cat((data,input))

        if count%1024==0:
            print("Count {}", count)
        count+=64


    print("Mean first channel : ", data[:,0,:,:].mean())
    print("Mean second channel : ",data[:,1,:,:].mean())
    print("Mean third channel : ", data[:,2,:,:].mean())

    print("Std first channel : ", data[:,0,:,:].std())
    print("Std second channel : ",data[:,1,:,:].std())
    print("Std third channel : ", data[:,2,:,:].std())


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
    
    elif dataset == "fashionmnist":
        print("Not Implemented") 
    
    elif dataset == "imagenet":
        
        traindir = os.path.join(root_directory, 'trainSample')
        valdir = os.path.join(root_directory, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        transformation = transforms.Compose([
                 transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
       

        train_dataset = torchvision.datasets.ImageFolder(traindir, transform = transformation)
    
#         
        train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        pin_memory=True)
        return train_loader, val_loader
    # elif dataste == ""

def GetAppropriateModel(arch, dataset, dumpData=False):
    if arch == "resnet18":
        if dataset == "cifar10" :
            print("Going for cifar10")
            return  resnet18(32, num_classes=10)
        elif dataset == "cifar100":
            print("Going for cifar100")
            return  resnet18(32, num_classes=100)
        
    elif arch == "resnet152":
        if dataset == "cifar10":
            print("Going for resnet152 cifar10")
            return resnet152(num_classes=10)
        elif dataset == "cifar100":
            print("Going for resnet152 cifar100")
            return resnet152(num_classes=100)
        
    elif arch == "resnet50":
        if dataset == "cifar10":
            print("Going for cifar10 resnet50")
            print("DumpData ",dumpData)
            return resnet50( num_classes=10)
        elif dataset == "cifar100":
            print("Going for cifar100 resnet50")
            return resnet50(num_classes=100)    
        
def GetAppropriateModelFreeze(arch, dataset, dumpData=False, dumpPath="emptyPath", dumpLayer=0):
    if arch == "resnet18":
        if dataset == "cifar10" :
            print("Going for cifar10")
            return  resnet18(32, num_classes=10, dumpData=dumpData, dumpPath=dumpPath, dumpLayer=dumpLayer)
        elif dataset == "cifar100":
            return  resnet18(32, num_classes=100, dumpData=dumpData, dumpPath=dumpPath, dumpLayer=dumpLayer)
    elif arch == "resnet50":
        if dataset == "cifar10" :
            print("Going for cifar10")
            return  resnet50(32, num_classes=10, dumpData=dumpData, dumpPath=dumpPath,dumpLayer=dumpLayer)
        elif dataset == "cifar100":
            print("Going for cifar100")
            return  resnet50(32, num_classes=100, dumpData=dumpData, dumpPath=dumpPath, dumpLayer=dumpLayer)
    elif arch == "resnet152":
        if dataset == "cifar10":
            print("Going for resnet152 cifar10")
            return resnet152(num_classes=10, dumpData=dumpData, dumpPath=dumpPath, dumpLayer=dumpLayer)
        elif dataset == "cifar100":
            print("Going for resnet152 cifar101")
            return resnet152(num_classes=100, dumpData=dumpData, dumpPath=dumpPath, dumpLayer=dumpLayer)
        
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
