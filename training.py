# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
import matplotlib as plt
from torch.utils.data import random_split


DEFAULT_DATASET = 'data'
DEFAULT_MODEL = 3  
DEFAULT_CUDA = "cuda:0"

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch): 
    print('Training on {} samples...'.format(len(train_loader.dataset)))  
    model.train()  
    for batch_idx, data in enumerate(train_loader): 
        data = data.to(device)   
        optimizer.zero_grad()  
        output = model(data)  
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))   
        loss.backward()  
        optimizer.step() 
        if batch_idx % LOG_INTERVAL == 0: 
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):  
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_dynamic_features = torch.Tensor()  
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_dynamic_features = torch.cat((total_dynamic_features, data.dynamic_features.cpu()), 0) 
        return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_dynamic_features.numpy() 


# datasets = [['data'][int(sys.argv[1])]]
# modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
# model_st = modeling.__name__

datasets = [DEFAULT_DATASET]
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][DEFAULT_MODEL]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('cuda_name:', cuda_name)

if torch.cuda.is_available():
    print('Number of available GPUs:', torch.cuda.device_count())
    print('GPU device name:', torch.cuda.get_device_name(0))  # 0表示第一个GPU设备
    print('CUDA version:', torch.version.cuda)
else:
    print('CUDA is not available. Training on CPU...')


TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 5 * 1e-4
LOG_INTERVAL = 20
NUM_EPOCHS = 500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


def rmse(y, f):
    return np.sqrt(mean_squared_error(y, f))
def pearson(y, f):
    return pearsonr(y, f)[0]

def evaluate_model(model, device, test_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_dynamic_features = torch.Tensor() 
    print('Make prediction for {} samples...'.format(len(test_loader.dataset)))
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            total_dynamic_features = torch.cat((total_dynamic_features, data.dynamic_features.cpu()), 0)  

    preds = total_preds.numpy().flatten()
    labels = total_labels.numpy().flatten()

    rmse_value = rmse(labels, preds)
    pr_value = pearson(labels, preds)

    return rmse_value, pr_value

for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset+'_train')
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')

        torch.manual_seed(42)
        total_size = len(train_data)
        val_size = int(0.2 * total_size)  
        train_size = total_size - val_size

        train_subset, val_subset = random_split(train_data, [train_size, val_size])

        train_loader = DataLoader(train_subset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=TRAIN_BATCH_SIZE, shuffle=False) 
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_rmse = 1e8
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
        result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        tmp = 0
        rmse_test = 0
        pr_test = 0
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1)
            rmse_value, pr_value = evaluate_model(model, device, val_loader)
            if rmse_value < best_rmse:
                best_epoch = epoch + 1
                torch.save(model.state_dict(), model_file_name)  
                print('RMSE improved at epoch ', best_epoch, '; Best RMSE:', rmse_value, '; Pr:',pr_value)
                tmp = pr_value
                best_rmse = rmse_value
                rmse_test, pr_test = evaluate_model(model, device, test_loader)
            elif rmse_value == best_rmse and tmp < pr_value:
                tmp = pr_value
                torch.save(model.state_dict(), model_file_name) 
                rmse_test, pr_test = evaluate_model(model, device, test_loader)
            else:
                print('No improvement since epoch ', best_epoch, '; Best RMSE:', best_rmse, '; Pr:',tmp)

        print('Best RMSE achieved at epoch {}: {:.6f}'.format(best_epoch, best_rmse))
        print('RMSE_test', rmse_test)
        print('Pr_test', pr_test)
