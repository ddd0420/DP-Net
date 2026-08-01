
import os
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
import sys
import argparse
import re
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from utils.loss import DiceLoss, SoftIoULoss, to_one_hot
from utils.losses import FocalLoss
from utils.Generate_Prototype import *
from dataloaders.dataset import *
from utils.train_util import *
from utils import test_3d_patch
from networks.seperate_vnet import VNet_D, VNet_E
from skimage.measure import label
gpus = [0]
dataset_dict = {
    "PAN": "../../dataset/Pancreas/",
    "LA": "../../dataset/LA/",
    "TBAD": "/root/autodl-tmp/dataset/TBAD/ImageTBAD",
    "BRA": "../../dataset/BraTS2019/"
}

def get_arguments():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--exp', type=str, default='test', help='experiment_name')
    parser.add_argument('--alpha', type=float, default=0.99, help='params in ema update')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--root_path', type=str, default='../results', help='Paths to previous checkpoints')

    # dataset
    parser.add_argument('--labeled_num', type=int, default=16, help='set the seed of random initialization')
    parser.add_argument('--dataset', type = str, default="LA")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/dataset/LA/2018LA_Seg_Training Set",
                        help="Path to the dataset.")
    parser.add_argument("--save_path", type=str, default='../results',
                        help="Path to save.")

    # Optimization options
    parser.add_argument('--lab_batch_size', type=int,  default=2, help='batch size')
    parser.add_argument('--unlab_batch_size', type=int,  default=2, help='batch size')
    parser.add_argument('--lr', type=float,  default=0.001, help='learning rate')
    parser.add_argument('--iters', type=int,  default=10000, help='maximum iter number to pretraining')
    parser.add_argument('--save_step', type=int,  default=200, help='frequecy of checkpoint save in pretraining')
    parser.add_argument('--consistency_rampup', type=float,
                        default=10000.0, help='consistency_rampup')

    parser.add_argument('--beta1', type=float,  default=0.5, help='params of optimizer Adam')
    parser.add_argument('--beta2', type=float,  default=0.999, help='params of optimizer Adam')
    parser.add_argument('--scaler', type=float,  default=1, help='multiplier of prototype')
    
    # Miscs
    parser.add_argument('--gpu', type=str,  default='-1', help='GPU to use')
    parser.add_argument('--seed', type=int, default=1230, help='set the seed of random initialization')
    return parser.parse_args()

args = get_arguments()

# Use CUDA
if int(args.gpu)>=0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
else:
    from search_to_run import run_gpu
    gpu_id = run_gpu(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

# create logger
resultdir = os.path.join(args.save_path, args.exp)
savedir = os.path.join(resultdir, 'inference')
os.makedirs(savedir, exist_ok=True)

data_dir = dataset_dict[args.dataset]

class infer_net():
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
    def __call__(self, *args, **kwargs):
        self.encoder.eval()
        self.decoder.eval()
        temp = self.encoder(*args, **kwargs)
        res = self.decoder(temp)
        self.encoder.train()
        self.decoder.train()
        return res
def main():
    save_path = Path(savedir)
    pth_path = os.path.join(resultdir,"checkpoints")
    data_dir = dataset_dict[args.dataset]

    encoder = VNet_E().cuda()
    checkpoint = torch.load(pth_path+"/best_encoder.pth")

    encoder.load_state_dict(checkpoint)
    linear_decoder = VNet_D(has_dropout=True).cuda()
    checkpoint = torch.load(pth_path+"/best_decoder.pth")
    linear_decoder.load_state_dict(checkpoint)

    net = infer_net(encoder, linear_decoder)
    test_all(net, data_dir, test_save_path=save_path)

def test_all(net, data_root, maxdice=0.0, save_result = True, test_save_path = None, num_classes=2):
    avg_metrics, std_metrics = test_3d_patch.var_all_case(net, data_root, save_result = True, test_save_path=test_save_path)
    val_dice = avg_metrics[0]
    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False

    print('Evaluation : val_dice: %.4f, val_maxdice: %.4f\n' % (val_dice, maxdice))
    
    print('\nDice:')
    print('Mean :%.2f(%.2f)' % (avg_metrics[0], std_metrics[0]))

    print('\nJaccard:')
    print('Mean :%.2f(%.2f)' % (avg_metrics[1], std_metrics[1]))

    print('\nHD95:')
    print('Mean :%.2f(%.2f)' % (avg_metrics[2], std_metrics[2]))

    print('\nASSD:')
    print('Mean :%.2f(%.2f)' % (avg_metrics[3], std_metrics[3]))
    
    return val_dice, maxdice, max_flag
if __name__ == '__main__':
    main()
