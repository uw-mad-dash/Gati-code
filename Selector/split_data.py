import os
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataDir', type=str, help='path to intermediate layer dumps.')
parser.add_argument('--layerNum', type=int, help='Layer to split.')

args = parser.parse_args()
def split(test_data, train_data, test_output_file, train_output_file, args):
    test_data_list = []
    filename = os.path.join(args.dataDir, test_data)
    index = 0
    with open(filename) as f:

        for line in f:
            if index > 1000: break
            test_data_list.append(line)
            index += 1
    index = 0
    with open(os.path.join(args.dataDir, train_data) ) as f:

        for line in f:
            if index > 2000: break
            test_data_list.append(line)
            index += 1


    with open(os.path.join(args.dataDir, test_output_file), "w") as f:
        for line in test_data_list:
            f.write(line )

    index = 0

    train_data_list = []
    with open(filename) as f:
        index = 0
        for line in f:
            if index < 1000: 
                index += 1
                continue
            train_data_list.append(line)
            index += 1
    with open(os.path.join(args.dataDir, train_data) ) as f:
        index = 0
        for line in f:
            if index < 2000: 
                index += 1
                continue
            train_data_list.append(line)
            index += 1

    with open(os.path.join(args.dataDir, train_output_file), "w") as f:
        for line in train_data_list:
            f.write(line)


split("Test_LayerOutput_block_{}".format(str(args.layerNum)), "Train_LayerOutput_block_{}".format(str(args.layerNum)), "Val_Test_LayerOutput_block_{}".format(str(args.layerNum)), "Val_Train_LayerOutput_block_{}".format(str(args.layerNum)), args)
split("Test_PredictedLabels", "Train_PredictedLabels", "Val_Test_PredictedLabels", "Val_Train_PredictedLabels", args)
