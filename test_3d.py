import os
import argparse
import torch
import pdb
import shutil

from networks.VNet import VNet
from networks.net_factory import net_factory
from utils.test_3d_patch_bcp import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='LA', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='LA', help='exp_name')
parser.add_argument('--model', type=str,  default='VNet', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-procssing?')
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--stage_name',type=str, default='train', help='train or pre_train')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "./results/{}/checkpoints/".format(FLAGS.exp)
test_save_path = "./results/{}/predictions/".format(FLAGS.exp)
if os.path.exists(test_save_path):
    shutil.rmtree(test_save_path)
os.makedirs(test_save_path)
print(test_save_path)

num_classes = FLAGS.num_classes

dataset_dict = {
    "PAN": "C:/PycharmProjects/dataset/Pancreas/",
    "LA": "C:/PycharmProjects/dataset/LA/",
    "TBAD": "/root/autodl-tmp/dataset/TBAD/ImageTBAD",
    "BRA": "C:/PycharmProjects/dataset/BraTS2019/",
    "ACDC": "C:/PycharmProjects/dataset/ACDC/",
    "PROMISE": "C:/PycharmProjects/dataset/Prostate/"
}
data_dir = dataset_dict[FLAGS.dataset]

if 'LA' in data_dir:
    with open(data_dir + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [data_dir + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    patch_size = (112, 112, 80)
    stride_xy = 18
    stride_z = 4

elif 'BRA' in data_dir:
    with open(data_dir + '/val.txt', 'r') as f:
        image_list = f.readlines()
    image_list = [data_dir + "/data/" + item.replace('\n', '') + ".h5" for item in image_list]
    patch_size = (96, 96, 96)
    stride_xy = 64
    stride_z = 64

elif "Pan" in data_dir:
    with open(data_dir + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [data_dir + "/Pancreas_h5/" + item.replace('\n', '') + ".h5" for item in image_list]
    patch_size = (96, 96, 96)
    stride_xy = 16
    stride_z = 16

def test_calculate_metric():
    model = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm').cuda()
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))

    model.eval()

    avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=patch_size, stride_xy=stride_xy, stride_z=stride_z,
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric

if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
