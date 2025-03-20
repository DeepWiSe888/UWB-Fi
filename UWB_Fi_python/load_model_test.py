#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load and test a trained model

"""

import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from args_det import args, device
# from models.dense_ed import DenseED
from models.dense_att_ed import DenseED
from utils.load_data import load_data
from utils.misc import mkdirs
from utils.plot import plot_prediction_det, save_stats
import json
import scipy.io
# from torchsummary import summary
from time import time

torch.cuda.empty_cache()

args.train_dir = args.run_dir + "/pre_result"
args.pred_dir = args.train_dir + "/predictions"


model = torch.load('/home/lixin/UWBFi/UWB-Fi_demo/experiments/deterministic/checkpoints/model_epoch100.pth')

# load data! change the path
test_data_dir = '/home/lixin/UWBFi/UWB-Fi_demo/data/save_path/testData.mat'

print('Loaded data!')

# test data
test_loader, test_stats = load_data(test_data_dir,1,False)
pre_result_t = []

model.eval()
mse = 0.
with torch.no_grad():
    for batch_idx, (input, target) in enumerate(test_loader):
        input, target = input.to(device), target.to(device)    
    
        output = model(input)
        mse += F.mse_loss(output, target, size_average=False).item()
        pre_result_t.append(output.data.cpu().numpy()) 


scipy.io.savemat(args.pred_dir + '/pre_result.mat',{'pre_result':pre_result_t})

