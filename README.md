# Gati: Accelerating Deep Learning Inference via Learned Caches

This is the code base for Gati, where we develop an end-to-end prediction serving system that incorporates early exit networks for low-latency DNN inference. The early exit networks contain the Predictor network, which takes in hidden layer outputs and produces the prediction early (mimicing the behavior of the base DNN), and the Selector network which decides whether we should use the output from the Predictor Network. The current code presents how to construct the Predictor and Selector Networks based on the hidden layer outputs with base models of ResNet-18/ResNet-50/ResNet-152 on CIFAR-10/CIFAR-100 datasets.   

### Requirements

+ pandas
+ scipy
+ torch==1.6.0
+ sklearn
+ torchvision



### Base model training

In Gati/Resnet/

```
python train_base.py \
--lr 0.01 \
--arch <resnet18/resnet50> \
--dataset <cifar10/cifar100>

```
### Dumping intermediate output using the trained model

In Gati/Resnet/:
```
python dump_data.py \
--dump_dir <directory to dump the intermediate output to > \
--dump_layer <layer number to dump the intermediate output > \
--resume \
--dataset_dir <Path to which the dataset is stored>
```


### Predictor 


We have three architectures for the Predictor Network: (i) Fully-connected (ii) Pooling (iii) Convolution

In Gati/Predictor/

##### (i) Fully-connected
```
python main.py \
--layerNum <layer to train> \
--baseCheckpoint <Path to base DNN model checkpoint> \
--trainMode \
--evalMode  \
--hiddenUnits <Number of hidden units> \
--saveModel \
--checkpointDir <Directory to save the trained Predictor network> \
--archType 1 \
--cifarDataDir <Path to the cifar datasets> \
--ioDir <Path where the intermediate outputs are stored> 
```


#### (ii) Pooling
```
python main.py \
--layerNum <layer to train> \
--baseCheckpoint <Path to base DNN model checkpoint> \
--trainMode \
--evalMode  \
--kernelSize <size of the kernel in the convolution network>  \
--stride <stride used in the conv network> \
--saveModel \
--checkpointDir <Directory to save the trained Predictor network> \
--archType 2  \
--classNum <number of classes in the base model> \
--cifarDataDir <Path to the cifar datasets> \
--ioDir <Path where the intermediate outputs are stored> 

```

#### (iii) Convolution
```
python main.py \
--layerNum <layer to train> \
--baseCheckpoint <Path to base DNN model checkpoint> \
--trainMode \
--evalMode \
--afterPoolingSize <after pooling size> \
--saveModel \
--checkpointDir <Directory to save the trained Predictor network> \
--archType 3 \
--cifarDataDir <Path to the cifar datasets> \
--ioDir <Path where the intermediate outputs are stored> \
--class_num <Number of classes > \
--lr 0.005
```

Different architectures of the Predictor Networks can achieve different accuracy, for example, we show the accuracy of Predictor networks for ResNet-18 on the test dataset for the CIFAR-10 dataset

|         | FC(1024) | Pool(4096) | Conv(3,1) |
|---------|----------|------------|-----------|
| Layer 1 | 55.14%   | 55.19%     | 57.63%    |
| Layer 2 | 60.68%   | 68.34%     | 66.04%    |
| Layer 3 | 78.64%   | 78.26%     | 75.91%    |
| Layer 4 | 85.34%   | 84.61%     | 82.64%    |
| Layer 5 | 94.5%    | 92.49%     | 93.28%    |
| Layer 6 | 99.19%   | 99.01%     | 98.28%    |
| Layer 7 | 99.95%   | 99.95%     | 99.94%    |
| Layer 8 | 100%     | 100%       | 100%      |

Also, we can get different performance (latency/memory cost) for different Predictor Network architectures. For example, for the 7th layer of the ResNet-18 model, we got the following latency and memory cost measurements on P100 GPU

|            | GPU latency(ms) | Memory Cost |
|------------|-----------------|-------------|
| FC(1024)   | 0.43            | 33MB        |
| Pool(4096) | 0.6             | 17MB        |
| Conv(3,1)  | 0.52            | 0.8MB       |




### Selector Network

In Gati/Selector/

#### To split data for training/testing Selector Network
python split_data.py --dataDir <Directory where intermediate outputs are stored. > --layerNum <Layer to split/train>
  
 #### Train Selector Network using trained Predictor Network and intermediate output
 ```
 python train.py \
 --trainMode \
 --evalMode \
 --DRcheckpoint <Path to the trained Predictor Network.> \
 --saveModel \
 --checkpointDir <Directory to store trained selector network.> \
 --layerNum <Layer to train> \
 --dataDir <Layer where training data for training Selector Network is stored.>
 ```
