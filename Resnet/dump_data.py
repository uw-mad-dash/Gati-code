'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--arch', default='resnet18', type=str, help='Type of model to train. ')
parser.add_argument('--dataset', default='cifar10', type=str, help='Type of dataset. ')
parser.add_argument('--dataset_dir', default='/mnt', type=str, help='Directory where dataset should be stored. ')
parser.add_argument('--dump_dir', default='/mnt/yuhanl', type=str, help='Directory where dataset should be stored. ')
parser.add_argument('--dump_layer', default=0, type=int, help='The layer to dump. ')

parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
                    
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
trainloader, testloader = GetTrainTestData(args.dataset, 128, dataset_dir=args.dataset_dir, shuffle_train = False)


# Label files
files = {}
files["labelsPrediction"] = open(os.path.join(args.dump_dir, "PredictedLabels"), "w")
files["labelsReal"] = open(os.path.join(args.dump_dir, "RealLabels"), "w")
# Model
print('==> Building model..')
# if os.path.exists( args.dump_dir) is False:
#     os.makedirs(args.dump_dir, exist_ok=True)
# elif os.listdir(args.dump_dir):
#     raise ValueError("Dump path not empty! ")

net = GetAppropriateModelFreeze(args.arch, args.dataset, dumpData=True, dumpPath = args.dump_dir, dumpLayer = args.dump_layer)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, FROZEN = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(test_dataloader):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, FROZEN = net(inputs)
            batchSize = outputs.size(0)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            if device == "cpu":
                np.savetxt(files["labelsPrediction"], predicted.view(batchSize, -1).numpy(), fmt='%1.1f', delimiter=',')
                np.savetxt(files["labelsReal"], targets.view(batchSize, -1).numpy(), fmt='%1.1f', delimiter=',')
            else:
                np.savetxt(files["labelsPrediction"], predicted.view(batchSize, -1).cpu().numpy(), fmt='%1.1f', delimiter=',')
                np.savetxt(files["labelsReal"], targets.view(batchSize, -1).cpu().numpy(), fmt="%1.1f", delimiter=',')
            test_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


# We dump intermediate data using training dataset 
test(trainloader)