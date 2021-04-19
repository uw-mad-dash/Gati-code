import os
import argparse 
import numpy as np
import torch, torchvision
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import pandas as pd
import utils
import models
import os 
import time
import torch.nn.functional as F
from Resnet import resnet18, resnet152, resnet50
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
            print("Going for resnet152 cifar101")
            return resnet152(num_classes=100)
       
    elif arch == "resnet50":
        if dataset == "cifar10":
            print("Going for cifar10 resnet50")
            print("DumpData ",dumpData)
            return resnet50( num_classes=10)
        elif dataset == "cifar100":
            print("Going for cifar100 resnet50")
            return resnet50(num_classes=100)    

def adjustLearningRate(optimizer, decay_factor):
    for param_group in optimizer.param_groups:
        prev = param_group['lr']
        param_group['lr'] = param_group['lr']*decay_factor
        print("Learning Rate Decayed {0} --> {1}".format(prev, param_group['lr']))
def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def train(model, epoch, criterion, optimizer, data, device, archType, base_model_outputs):
    model.train()
    print("Boss, Train Cap On!")
    total = 0
    correct = 0
    for batchNum, (inputs, labels) in enumerate(data):
        
        inputs = inputs.to(device).float()
        
        if archType == 3 or archType == 2:
            
            inputs = inputs.view((inputs.shape[0], 1, inputs.shape[1]))
        labels = labels.to(device).long()
        
        outputs = model(inputs)
        
        base_outputs = base_model_outputs[(batchNum * args.batchSize): (batchNum + 1) * args.batchSize]


        input_array_np = np.array(base_outputs)
        base_outputs_tensor = torch.Tensor(input_array_np)
        
        base_outputs_tensor = base_outputs_tensor.to(device).float()
        loss = loss_fn_kd(outputs, labels, base_outputs_tensor, args.alpha, args.temperature)
        # zero the gradient
        optimizer.zero_grad()
        # calculate the gradients
        loss.backward()
        # take the step by the optuimizer
        optimizer.step()
        _, predicted = outputs.max(1)
        batchSize = outputs.size(0)
        correctBatch = (predicted==labels).sum().item()    
        total = total + batchSize
        correct = correct + correctBatch
    
        if batchNum%10==0:
            print("Epoch:{0}, BatchNum:{1}, Loss:{2}, Accuracy : {3}".format(epoch, batchNum, loss, correct*1.0/total))
        



def test(model, epoch, criterion, data, device,max_batch=10000,archType=1):
    # eval mode
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for batchNum, (input, targets) in enumerate(data):
            input, targets = input.to(device), targets.to(device)
            input = input.float()
            if archType == 3 or archType == 2:
            
                input = input.view((input.shape[0], 1, input.shape[1]))
            targets  = targets.long()
            output = model(input)
            
            _, predicted = output.max(1)
            batchSize = output.size(0)
            correctBatch = (predicted==targets).sum().item()
            total = total + batchSize
            correct = correct + correctBatch
            targets_np = targets.cpu().numpy()
            
            if batchNum%10==0:
                print("Epoch: {0} Accuracy: {1}, TotalEx: {2}, CorrectEx: {3} ".format(epoch, correct*1.0/total, total, correct))
        return correct*1.0/total 
def base_model_predict(dataGen, model, criterion, device):
    model.eval()
    total = 0
    correct = 0
    acc = 0
    points = 0
    overallAcc = 0
    base_model_predicted = []
    base_model_outputs = []
    with torch.no_grad():
        for batchNum, (input, targets) in enumerate(dataGen):
            input, targets = input.to(device), targets.to(device)
            start = time.time()
            output, FROZEN = model(input)
            base_model_outputs.append(output)
            
            loss = criterion(output, targets)
            _, predicted = output.max(1)
            batchSize = output.size(0)
            end=time.time()
            base_model_predicted.append(predicted)
            correctBatch = (predicted==targets).sum().item()
            total = total + batchSize
            correct = correct + correctBatch
            
    return base_model_outputs, base_model_predicted


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--archType', default=1, type=int, help='type of architecture')
    parser.add_argument('--cifarDataDir', type=str, default='/mnt/')
    parser.add_argument('--ioDir', type=str, default='/mnt/yuhanl', help='Path where the intermediate output is stored. ')
    parser.add_argument('--trainMode', action='store_true', help='train mode')
    parser.add_argument('--evalOnTrainData', action='store_true', help='train mode')
    parser.add_argument('--evalMode',  action='store_true', help='test mode')
    parser.add_argument('--numEpochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--hiddenUnits', type=int, default=1024, help='hidden units for predictor type 1')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--decay', '--decay-factor', default=0.5, type=float, metavar='LR', help='leraning rate decay factor')
    parser.add_argument('--decay_after_n', type=int, default=5)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--batchSize', default=64, help='batch size')
    parser.add_argument('--checkpointDir', default='/data/checkpoints/DR/resnet152', type=str)
    parser.add_argument('--checkpointInterval', default=1, type=int)
    parser.add_argument('--saveModel', action='store_true')
    parser.add_argument('--resume', help='whether to resume from the path')
    parser.add_argument('--layerNum', type=int, help='index of layer for which data is used')
    parser.add_argument('--stride', type=int, help='stride for the conv1d arch.', default=5)
    parser.add_argument('--num_workers', type=int, help='num workers', default=1) 
    parser.add_argument('--kernelSize', type=int, help='kernel size of the 1d Convolution', default=3) 
    parser.add_argument('--afterPoolingSize', type=int, default=4096)
    parser.add_argument('--hiddenLayerSize', type=int, default=1024)
    parser.add_argument('--baseCheckpoint', type=str, default='')
    parser.add_argument('--alpha',  default=0.5, type=float, help='Alpha used in the knowledge distillation. ')
    parser.add_argument('--temperature', default=6, type=float,  help='Temperature used in the knowledge distillation. ')
    parser.add_argument('--classNum', default=10, type=float,  help='Number of classes. ')
    parser.add_argument('--baseArch', default='resnet18', type=str,  help='Arch of the base model. ')
    parser.add_argument('--dataset', default='cifar10', type=str,  help='Dataset to use. ')

    args = parser.parse_args()

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base_model = GetAppropriateModel(args.baseArch, args.dataset)
    base_model.to(device)

    
    if device == 'cuda':
        base_model = torch.nn.DataParallel(base_model)
    cifar_data_train, _ = utils.GetTrainTestData(args.dataset, args.batchSize, args.cifarDataDir, shuffle_train=False)

    base_model_checkpoint_ = torch.load(args.baseCheckpoint)
    
    base_model.load_state_dict(base_model_checkpoint_['net'])
    trainDataPath   = os.path.join(args.ioDir, "Train_LayerOutput_block_{}".format(str(args.layerNum)))
    testDataPath    = os.path.join(args.ioDir, "Test_LayerOutput_block_{}".format(str(args.layerNum)))
    
    testLabelsPath  = os.path.join(args.ioDir, "Test_PredictedLabels")
    trainLabelsPath = os.path.join(args.ioDir, "Train_PredictedLabels")
    trainData    = utils.DataNoNumpy(trainDataPath, trainLabelsPath)
    trainDataGen = DataLoader(trainData, batch_size=args.batchSize, num_workers=args.num_workers, shuffle=False)

   
    
    testData  = utils.DataNoNumpy(testDataPath, testLabelsPath)

    testDataGen = DataLoader(testData, batch_size=args.batchSize, num_workers=args.num_workers, shuffle=False)
    max_batch = len(trainDataGen)
    
    if args.archType == 1:
        model = models.ClassifierLayer(trainData.inputDim(), int(args.classNum), hiddenSize=args.hiddenUnits) 
    elif args.archType == 2: 
        model = models.PoolingClassifier(trainData.inputDim(), int(args.classNum), afterPoolingSize =args.afterPoolingSize, hiddenLayerSize = args.hiddenLayerSize)
    elif args.archType == 3:
        model = models.ConvClassifier(trainData.inputDim(), int(args.classNum), kernel_size=args.kernelSize, stride=args.stride)
    model.to(device)

    if args.resume:
        if device.type =="cpu": 
            checkpoint_ = torch.load(args.resume, map_location = device.type)
        else:
            checkpoint_ = torch.load(args.resume, map_location = device.type + ":" + str(device.index))

        best_acc = checkpoint_["best_acc"]
        model.load_state_dict(checkpoint_['state_dict'])
        epoch = checkpoint_['epoch']
    
    
    criterion = nn.CrossEntropyLoss()
    # Get the base model outputs, used for knowledge distillation
    base_model_outputs_train,base_model_predicted = base_model_predict(cifar_data_train, base_model, criterion, device)
    
    
    flatten_outputs = []
    flatten_predicted = []
    for output in base_model_outputs_train:
        
        for i in range(output.shape[0]):
            flatten_outputs.append(output[i].cpu().numpy())
            

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
    best_acc = 0
    if args.trainMode:
        print("Training Mode !")
        for epoch in range(1, args.numEpochs):
            if epoch%args.decay_after_n==0:
                adjustLearningRate(optimizer, args.decay) 
            

            train(model, epoch, criterion, optimizer, trainDataGen, device, archType = args.archType, base_model_outputs = flatten_outputs)
            #test(model, epoch, criterion, trainDataGenForInfer, device, max_batch=200)
            acc = test(model, epoch, criterion, testDataGen, device, archType = args.archType)

            # if acc better that best or regular interval, then save model
            if args.saveModel:
                filename = os.path.join(args.checkpointDir, 'd_checkpoint_{0}_arch_{1}.pth.tar'.format(args.layerNum, args.archType))
                if args.archType == 3:
                    best_acc_filename = os.path.join(args.checkpointDir, 'checkpoint_best_acc_layer_{0}_arch_{1}_stride_{2}_kernel_{3}_alpha_{4}_temperature_{5}.pth.tar'.format(args.layerNum, args.archType, args.stride, args.kernelSize, args.alpha, args.temperature))
                elif args.archType == 2:
                    best_acc_filename = os.path.join(args.checkpointDir, 'checkpoint_best_acc_layer_{0}_arch_{1}_ps_{2}_hs_{3}_alpha_{4}_temperature_{5}.pth.tar'.format(args.layerNum, args.archType, args.afterPoolingSize, args.hiddenLayerSize, args.alpha, args.temperature))
                else:
                    best_acc_filename = os.path.join(args.checkpointDir, 'checkpoint_best_acc_layer_{0}_arch_{1}_hs_{2}_alpha_{3}_temperature_{4}.pth.tar'.format(args.layerNum, args.archType, args.hiddenUnits, args.alpha, args.temperature))

                
                print("Best Acc Till Now : {0}, Acc this epoch : {1}".format(best_acc, acc))
                
                state = {
                        'epoch': epoch+1,
                        'acc': acc,
                        'best_acc': best_acc,
                        'state_dict': model.state_dict(),
                        }
            
                # every regular interval save
                if epoch%args.checkpointInterval==0:
                    torch.save(state, filename)
                # if best till now
                if acc>best_acc:
                    best_acc = acc
                    torch.save(state, best_acc_filename)
    
    elif args.evalMode:
        if args.evalOnTrainData:
            test(model,None, epoch, criterion, trainDataGen, device, args, archType = args.archType)
        else :

            test(model,None, epoch, criterion, testDataGen, device, args, archType=args.archType)


