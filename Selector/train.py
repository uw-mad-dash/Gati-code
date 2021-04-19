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
import model
import os 
import time
import math
from sklearn.metrics import log_loss
from torch.utils.data.sampler import SubsetRandomSampler

torch.manual_seed(0)
# python main.py --layerNum 13 --DRcheckpoint /data/checkpoints/DR/resnet50/checkpoint_best_acc_layer_13_arch_1.pth.tar --dataDir /split/balarjun/RESNET50/TrainData/ --trainMode --saveModel --checkpointDir /data/checkpoints/SELECTOR/

def adjustLearningRate(optimizer, decay_factor):
    for param_group in optimizer.param_groups:
        prev = param_group['lr']
        param_group['lr'] = param_group['lr']*decay_factor
        print("Learning Rate Decayed {0} --> {1}".format(prev, param_group['lr']))



def train(model, epoch, criterion, optimizer, data, device, alpha=0.5):
    model.train()
    print("Boss, Train Cap On!")
    total = 0
    correct = 0
    for batchNum, (inputs, labels) in enumerate(data):
        
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        # zero the gradient
        optimizer.zero_grad()
        # calculate the gradients
        loss.backward()
        # take the step by the optuimizer
        optimizer.step()
        _, predicted = outputs.max(1)
        batchSize = outputs.size(0)
        
        correctBatch = (predicted==labels).sum().item()   
        softmax_scores = outputs.cpu().detach().numpy()
        pos_softmax_scores = softmax_scores[:, 1]
        thres_array = np.full((outputs.shape[0]), alpha)
        threshold_predicted = np.all([pos_softmax_scores > alpha], 0)
        threshold_predicted = threshold_predicted.astype(int)
        total = total + batchSize
        correct = correct + correctBatch
        
        if batchNum%10==0:
            print("Epoch:{0}, BatchNum:{1}, Loss:{2}, Accuracy : {3}".format(epoch, batchNum, loss, correct*1.0/total))
            
    return

def test(model, epoch, data, criterion, device,max_batch=10000, alpha=0.5):
    # eval mode
    model.eval()
    total = 0
    correct = 0
    
    
    with torch.no_grad():
        TP = 0
        FN = 0
        FP = 0
        epoch_precision = 0
        epoch_recall = 0
        for batchNum, (input, targets) in enumerate(data):
            input, targets = input.to(device), targets.to(device)
            input = input.float()
            targets  = targets.long()
            output = model(input)
            loss = criterion(output, targets)
            _, predicted = output.max(1)
            batchSize = output.size(0)
            correctBatch = (predicted==targets).sum().item()
            total = total + batchSize
            correct = correct + correctBatch
            predicted_np = predicted.cpu().numpy()
            output_np = output.cpu().numpy()
            targets_np = targets.cpu().numpy()

            softmax_scores = output.cpu().numpy()
            pos_softmax_scores = softmax_scores[:, 1]
            thres_array = np.full((output.shape[0]), alpha)
            threshold_predicted = np.all([pos_softmax_scores > alpha], 0)
            threshold_predicted = threshold_predicted.astype(int)
            

            TP += np.all([targets_np==threshold_predicted,threshold_predicted==1], 0).sum()
            FN += np.all([targets_np!=threshold_predicted,threshold_predicted==0, targets_np==1], 0).sum()
            FP += np.all([targets_np!=threshold_predicted,threshold_predicted==1, targets_np==0], 0).sum()
#             print("TP: {0}, FN: {1}, FP: {2}".format(TP, FN, FP))
            if batchNum>max_batch:
                break
        if TP + FP > 0 and TP + FN > 0:
            epoch_precision = TP /(TP + FP)
            epoch_recall = TP/(TP + FN) 
        
        
        return correct*1.0/total, epoch_precision, epoch_recall
    
def predict(dr_model, trainDataGen, DRcheckpoint, device, archType):
    """ Generates confidence scores using trained DR models. 
    """
    dr_model.to(device)
    if device.type =="cpu": 
        checkpoint_ = torch.load(DRcheckpoint, map_location = device.type)
    else:
        checkpoint_ = torch.load(DRcheckpoint, map_location = device.type + ":" + str(device.index))
    best_acc = checkpoint_["best_acc"]
    # model loaded
    dr_model.load_state_dict(checkpoint_['state_dict'])
    epoch = checkpoint_['epoch']
    criterion = nn.CrossEntropyLoss()
    dr_model.eval()
    predicted_labels = [] #np.array([])
    score_vectors = [] #np.array([])
    total = 0
    correct = 0
    target_labels = []
    with torch.no_grad():
        
        for batchNum, (input, targets) in enumerate(trainDataGen):
            print("batch: ", batchNum)
            input, targets = input.to(device), targets.to(device)
            input = input.float()
            if archType == 3 or archType == 2:
            
                input = input.view((input.shape[0], 1, input.shape[1]))
            
            targets  = targets.long()
            output = dr_model(input)
            _, predicted = output.max(1)
            np_prob = output.cpu().numpy()
            np_predicted = predicted.cpu().numpy()
            correctBatch = (predicted==targets).sum().item()
            batchSize = np_predicted.shape[0]
            total = total + batchSize
            targets = targets.cpu().numpy()
            correct = correct + correctBatch
            for i in range(np_prob.shape[0]):
                predicted_labels.append(np_predicted[i])
                score_vectors.append(np_prob[i])
                target_labels.append(targets[i])
                

    print("Accuracy: {0}".format(correct / total))
    return np.array(predicted_labels), np.array(score_vectors), np.array(target_labels)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataDir', type=str, help='path to intermediate layer dumps.')
    parser.add_argument('--DRcheckpoint', type=str, help='path to the DR trained model checkpoints.')
    parser.add_argument('--layerNum', type=int, help='Layer index.')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--decay', '--decay-factor', default=0.5, type=float, metavar='LR', help='leraning rate decay factor')
    parser.add_argument('--decay_after_n', type=int, default=5)
    parser.add_argument('--resume', type=str, help='whether to resume from the path')
    parser.add_argument('--trainMode', action='store_true', help='train mode')
    parser.add_argument('--evalOnTrainData', action='store_true', help='train mode')
    parser.add_argument('--evalMode',  action='store_true', help='test mode')
    parser.add_argument('--saveModel',  action='store_true', help='whether to save the checkpoints or not')
    parser.add_argument('--checkpointInterval', default=1, type=int)
    parser.add_argument('--numEpochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--checkpointDir', type=str, default='/data/checkpoints/SELECTOR/', help='path to save selector model checkpoints.')
    parser.add_argument('--archType', type=int, default=1, help='number of epochs')
    parser.add_argument('--train_size', type=int, default=50000, help='number of epochs')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha in the customized loss function.')
    parser.add_argument('--kernelSize', type=int, help='kernel size of the 1d Convolution', default=10) 
    parser.add_argument('--afterPoolingSize', type=int, default=4096)
    parser.add_argument('--hiddenLayerSize', type=int, default=1024)
    parser.add_argument('--hiddenUnits', type=int, default=1024, help='hidden size of dr model type 1')
    parser.add_argument('--stride', type=int, default=20)
    parser.add_argument('--classNum', type=int, default=10)
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--kd_alpha', default=0.5, type=float, help='alpha in the customized loss function.')
    parser.add_argument('--kd_temp', default=6, type=float, help='alpha in the customized loss function.')

    args = parser.parse_args()
    trainDataPath   = os.path.join(args.dataDir, "Val_Train_LayerOutput_block_{}".format(str(args.layerNum)))
    trainLabelsPath = os.path.join(args.dataDir, "Val_Train_PredictedLabels")

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    trainData = utils.DataNoNumpy(trainDataPath, trainLabelsPath)
    allDataGen = DataLoader(trainData, batch_size=128)
    
    
    print("done preprocessing {0}!".format(len(allDataGen)))
    args.archType = int(args.DRcheckpoint.split("_")[6])
    if args.archType == 1:
        args.hiddenUnits = int(args.DRcheckpoint.split("_")[8])
        dr_model = model.Classifier2Layer(trainData.inputDim(), args.classNum, hiddenSize=args.hiddenUnits) 
    elif args.archType == 2: 
        args.afterPoolingSize = int(args.DRcheckpoint.split("_")[8])
        args.hiddenLayerSize = int(args.DRcheckpoint.split("_")[10])
        dr_model = model.PoolingClassifier(trainData.inputDim(), args.classNum, afterPoolingSize =args.afterPoolingSize, hiddenLayerSize = args.hiddenLayerSize)
    elif args.archType == 3:
        args.stride = int(args.DRcheckpoint.split("_")[8])
        args.kernelSize = int(args.DRcheckpoint.split("_")[10])
        dr_model = model.ConvClassifier(trainData.inputDim(), args.classNum, stride=args.stride, kernel_size=args.kernelSize)
    else:
        raise ValueError("archType not supported! ")
    predicted_labels, score_vectors,gt_train_labels = predict(dr_model, allDataGen, args.DRcheckpoint, device, archType = args.archType)
    
    
    dr_acc = np.all([gt_train_labels==predicted_labels], 0).sum()
    correct_indices = np.all([gt_train_labels==predicted_labels], 0)
    incorrect_indices = np.all([gt_train_labels!=predicted_labels], 0)
    
        
    train_data_points = args.train_size
    neg_data_points = len(incorrect_indices)
    
    incorrect_vectors = score_vectors[incorrect_indices][:neg_data_points]
    incorrect_predicted_labels = predicted_labels[incorrect_indices][:neg_data_points]
    incorrect_gt_labels = gt_train_labels[incorrect_indices][:neg_data_points]


    correct_score_vectors = score_vectors[correct_indices][:train_data_points]
    correct_predicted_labels = predicted_labels[correct_indices][:train_data_points]
    correct_gt_labels = gt_train_labels[correct_indices][:train_data_points]    
    print("correct score vector: ", correct_score_vectors.shape)
    print("incorrect score vector: ", incorrect_vectors.shape)
        # prepare the whole dataset after down sampling
    all_score_vectors = np.concatenate((correct_score_vectors,incorrect_vectors))
    all_predicted_labels = np.concatenate((correct_predicted_labels, incorrect_predicted_labels))
    all_gt_labels = np.concatenate((correct_gt_labels, incorrect_gt_labels))
    
    
        
        
    val_data_points = int(0.2 * len(all_score_vectors))
    # prepare data for selector model
    dataset_size = len(all_score_vectors)
    indices = list(range(dataset_size))
    random_seed = 0
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[val_data_points:], indices[:val_data_points]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    
    selectorAllData = utils.DrData(all_score_vectors, all_predicted_labels, all_gt_labels)
    selectorTrainGen = DataLoader(selectorAllData, batch_size=128, num_workers=1, sampler=train_sampler)
    selectorTestGen = DataLoader(selectorAllData, batch_size=128, num_workers=1, sampler=valid_sampler)
    
    selector_model = model.BinaryClassifier(selectorAllData.inputDim(), 2) 
    
    selector_model.to(device)
    if args.resume:
        if device.type =="cpu": 
            checkpoint_ = torch.load(args.resume, map_location = device.type)
        else:
            checkpoint_ = torch.load(args.resume, map_location = device.type + ":" + str(device.index))

        best_acc = checkpoint_["best_acc"]
        selector_model.load_state_dict(checkpoint_['state_dict'])
        epoch = checkpoint_['epoch']
    
    optimizer = optim.SGD(selector_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
    criterion = nn.CrossEntropyLoss()
    max_precision = 0
    recall_at_max_precision = 0
    best_acc = 0
    if args.trainMode:
        print("Training Mode !")
        for epoch in range(1, args.numEpochs):
            if epoch%args.decay_after_n==0:
                adjustLearningRate(optimizer, args.decay) 

            train(selector_model, epoch, criterion, optimizer, selectorTrainGen, device, alpha=args.alpha)
            acc, epoch_precision, epoch_recall = test(selector_model, epoch, selectorTestGen, criterion, device, alpha=args.alpha)
            if epoch_precision > max_precision:
                max_precision = epoch_precision
                recall_at_max_precision = epoch_recall
                # if acc better that best or regular interval, then save model
            if args.saveModel:
                filename = os.path.join(args.checkpointDir, 'd_checkpoint_{0}_{1}.pth.tar'.format(args.layerNum, args.archType))
                if args.archType == 3:
                    best_acc_filename = os.path.join(args.checkpointDir, 'checkpoint_best_acc_layer_{0}_arch_{1}_stride_{2}_kernel_{3}_ts{4}_alpha_{5}_temperature_{6}.pth.tar'.format(args.layerNum, args.archType, args.stride, args.kernelSize, args.train_size, args.kd_alpha, args.kd_temp))
                elif args.archType == 2:
                    best_acc_filename = os.path.join(args.checkpointDir, 'checkpoint_best_acc_layer_{0}_arch_{1}_ps_{2}_hs_{3}_ts{4}_alpha_{5}_temperature_{6}.pth.tar'.format(args.layerNum, args.archType, args.afterPoolingSize, args.hiddenLayerSize, args.train_size, args.kd_alpha, args.kd_temp))
                else:
                    best_acc_filename = os.path.join(args.checkpointDir, 'checkpoint_best_acc_layer_{0}_arch_{1}_hs_{2}_ts{3}_alpha_{4}_temperature_{5}.pth.tar'.format(args.layerNum, args.archType, args.hiddenUnits, args.train_size, args.kd_alpha, args.kd_temp))
                
                print("Best Acc Till Now : {0}, Acc this epoch : {1}".format(best_acc, acc))

                state = {
                            'epoch': epoch+1,
                            'acc': acc,
                            'best_acc': best_acc,
                            'state_dict': selector_model.state_dict(),
                            }

                    # every regular interval save
                if epoch%args.checkpointInterval==0:
                    torch.save(state, filename)
                    # if best till now
                if acc>best_acc:
                    best_acc = acc
                    torch.save(state, best_acc_filename)
        print("Max precision and recall at the max precision: ", max_precision, recall_at_max_precision)
    elif args.evalMode:
        max_precision = 0
        recall_at_max_precision = 0
        if args.evalOnTrainData:
            acc, epoch_precision, epoch_recall = test(selector_model, epoch, selectorTrainGen, criterion, device, alpha=args.alpha)
            if epoch_precision > max_precision:
                max_precision = epoch_precision
                recall_at_max_precision = epoch_recall
        elif args.evalOnValData :
            acc, epoch_precision, epoch_recall = test(selector_model, epoch, selectorTestGen, criterion, device, alpha=args.alpha)
            if epoch_precision > max_precision:
                max_precision = epoch_precision
                recall_at_max_precision = epoch_recall
        
        print("Max precision and recall at the max precision: ", max_precision, recall_at_max_precision)        
        
