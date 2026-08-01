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
from torchvision import transforms

from utils.loss import DiceLoss, SoftIoULoss, to_one_hot
from utils.losses import FocalLoss
from utils.Generate_Prototype import *
from dataloaders.dataset import *
from utils.train_util import *
from utils import test_3d_patch
from networks.seperate_unet import UNet_D, UNet_E
from skimage.measure import label
from utils import val_2d

dataset_dict = {
    "PAN": "C:/PycharmProjects/dataset/Pancreas/",
    "LA": "C:/PycharmProjects/dataset/LA/",
    "TBAD": "/root/autodl-tmp/dataset/TBAD/ImageTBAD",
    "BRA": "C:/PycharmProjects/dataset/BraTS2019/",
    "ACDC": "C:/PycharmProjects/dataset/ACDC/",
    "PROMISE": "C:/PycharmProjects/dataset/Prostate/"
}

def get_arguments():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
    # parser.add_argument('--exp', type=str, default='la', help='experiment_name')
    parser.add_argument('--alpha', type=float, default=0.99, help='params in ema update')
    parser.add_argument('--resume', action='store_true')
    # dataset
    parser.add_argument('--labeled_num', type=int, default=7, help='set the seed of random initialization')
    parser.add_argument('--dataset', type=str, default="ACDC")
    parser.add_argument("--save_path", type=str, default='./results', help="Path to save.")
    parser.add_argument("--list_dir", type=str, default='C:/PycharmProjects/dataset/ACDC/datalist/ACDC',
                        help="datalist")
    # Optimization options
    parser.add_argument('--lab_batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--unlab_batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--iters', type=int, default=30000, help='maximum iter number to pretraining')
    parser.add_argument('--save_step', type=int, default=500, help='frequecy of checkpoint save in pretraining')
    parser.add_argument('--consistency_rampup', type=float, default=10000.0, help='consistency_rampup')

    parser.add_argument('--beta1', type=float, default=0.5, help='params of optimizer Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='params of optimizer Adam')
    parser.add_argument('--scaler', type=float, default=1, help='multiplier of prototype')
    parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
    # Miscs
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--seed', type=int, default=1337, help='set the seed of random initialization')
    return parser.parse_args()


args = get_arguments()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# create logger
resultdir = "{}/{}no_grad_bs{}_{}_labeled/".format(args.save_path, args.dataset, args.lab_batch_size, args.labeled_num)
# resultdir = os.path.join(args.save_path, args.exp)
logdir = os.path.join(resultdir, 'logs')
savedir = os.path.join(resultdir, 'checkpoints')
shotdir = os.path.join(resultdir, 'snapshot')
print('Result path: {}\nLogs path: {}\nCheckpoints path: {}\nSnapshot path: {}'.format(resultdir, logdir, savedir,
                                                                                       shotdir))

os.makedirs(logdir, exist_ok=True)
os.makedirs(savedir, exist_ok=True)
os.makedirs(shotdir, exist_ok=True)

writer = SummaryWriter(logdir)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)

fh = logging.FileHandler(shotdir + '/' + 'snapshot.log', encoding='utf8')
fh.setFormatter(formatter)
logger.addHandler(fh)
logging.info(str(args))
data_dir = dataset_dict[args.dataset]


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def proto_align_loss(proto1, proto2):
    loss = 0.0
    for c in range(args.num_classes):
        dot_product = torch.dot(proto1[c].view(-1), proto2[c].view(-1))
        norm_tensor1 = torch.norm(proto1[c])
        norm_tensor2 = torch.norm(proto2[c])
        loss = loss + 1 - (dot_product / (norm_tensor1 * norm_tensor2))
    return loss


def main():
    save_path = Path(savedir)

    set_random_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    data_dir = dataset_dict[args.dataset]

    encoder = UNet_E().cuda()
    decoder = UNet_D(has_dropout=True).cuda()

    optimizer_encoder = optim.Adam(encoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_linear_decoder = optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    args.root_path = data_dir

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))

    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    # db_val = BaseDataSets(base_dir=args.root_path, split="test")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                          args.lab_batch_size + args.unlab_batch_size,
                                          args.unlab_batch_size)

    train_loader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    dice_loss = DiceLoss(nclass=args.num_classes)
    ce_loss = CrossEntropyLoss()
    focal_loss = FocalLoss()
    iou_loss = SoftIoULoss(nclass=args.num_classes)
    pixel_level_ce_loss = CrossEntropyLoss(reduction='none')

    best_dice = 0.0
    iter_num = 0
    best_performance = 0
    patch_size = args.patch_size

    epoch_num = 1 + args.iters / len(train_loader)
    for _ in tqdm(range(int(epoch_num)), ncols=70):
        logging.info('\n')
        encoder.train()
        decoder.train()
        for _, sampled_batch in enumerate(train_loader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            iter_num += 1
            lab_img, lab_lab = volume_batch[:args.lab_batch_size].cuda(), label_batch[:args.lab_batch_size].cuda()
            unlab_img = volume_batch[args.lab_batch_size:].cuda()

            '''Strong Augmentation'''
            strategy = iter_num % 2
            color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            random_color_jittor = transforms.RandomApply([color_jitter], p=1.0)
            if strategy:
                aug_unlab = random_color_jittor(unlab_img)
            else:
                random_prob = torch.rand_like(unlab_img).cuda()
                mask = random_prob > 0.9
                aug_unlab = unlab_img * mask

            concat_img = encoder(torch.cat((lab_img, unlab_img, aug_unlab), dim=0))

            feat_u = []
            feat_l = []
            feat_u_aug = []
            for item in concat_img:
                feat_l.append(item.chunk(3)[0])
                feat_u.append(item.chunk(3)[1])
                feat_u_aug.append(item.chunk(3)[2])

            '''Feature perturbation'''
            feature_perb = [nn.Dropout2d(0.5)(feat) for feat in feat_u]
            feat_concat = [torch.cat(item, dim=0) for item in zip(feat_l, feat_u, feat_u_aug, feature_perb)]
            out_concat = decoder(feat_concat, False)
            out_l_linear, out_u, out_u_aug, out_u_fp = out_concat.chunk(4)

            # 1 decoder sup
            lab_lab_int64 = lab_lab.type(torch.int64)
            lab_lab_onehot = to_one_hot(lab_lab_int64.unsqueeze(1), args.num_classes)

            # labeled prototypes
            lab_fts = F.interpolate(feat_l[4], size=lab_lab.shape[-2:], mode='bilinear', align_corners=True)
            lab_prototypes = getPrototype2d(lab_fts, lab_lab_onehot)
            lab_fts_low = F.interpolate(feat_l[1], size=lab_lab.shape[-2:], mode='bilinear', align_corners=True)
            lab_prototypes_low = getPrototype2d(lab_fts_low, lab_lab_onehot)

            # 2 unlab proto dist
            '''Unsupervised'''
            with torch.no_grad():
                pred_u = torch.softmax(out_u, dim=1)
                uncertainty = -torch.sum(pred_u * torch.log(pred_u + 1e-16), dim=1)
                norm_uncertainty = torch.stack([uncertain / torch.sum(uncertain) for uncertain in uncertainty], dim=0)
                reliability_map = (1 - norm_uncertainty) / np.prod(np.array(norm_uncertainty.shape[-3:]))

                mask = torch.argmax(pred_u, dim=1)
                mask_onehot = to_one_hot(mask.unsqueeze(1), args.num_classes) # torch.Size([2, 4, 256, 256])
                unlab_fts = F.interpolate(feat_u[4], size=lab_lab.shape[-2:], mode='bilinear', align_corners=True)
                unlab_prototypes = getPrototype2d(unlab_fts, mask_onehot, reliability_map)
                unlab_fts_low = F.interpolate(feat_u[1], size=lab_lab.shape[-2:], mode='bilinear', align_corners=True)
                unlab_prototypes_low = getPrototype2d(unlab_fts_low, mask_onehot, reliability_map)

            pred_u_aug = torch.softmax(out_u_aug, dim=1)
            pred_u_fp = torch.softmax(out_u_fp, dim=1)

            loss_ce_ln = ce_loss(out_l_linear, lab_lab.long())
            loss_dice_ln = dice_loss(out_l_linear, lab_lab)
            loss_focal_ln = focal_loss(out_l_linear, lab_lab.type(torch.int64))
            loss_iou_ln = iou_loss(out_l_linear, lab_lab)
            ln_sup_loss = (loss_ce_ln + loss_dice_ln + loss_focal_ln + loss_iou_ln) / 4

            consistency_weight = get_current_consistency_weight(iter_num, args.consistency_rampup)

            '''Prototype fusion (lab + unlab)'''
            prototypes = [(lab_prototypes[c] + consistency_weight * unlab_prototypes[c]) / (1 + consistency_weight)
                          for c in range(args.num_classes)]
            lab_dist = torch.stack([calDist2d(lab_fts, prototype, scaler=args.scaler) for prototype in prototypes],
                                   dim=1)
            unlab_dist = torch.stack([calDist2d(unlab_fts, prototype, scaler=args.scaler) for prototype in prototypes],
                                     dim=1)

            prototypes_low = [(lab_prototypes_low[c] + consistency_weight * unlab_prototypes_low[c]) / (1 + consistency_weight)
                for c in range(args.num_classes)]
            lab_dist_low = torch.stack([calDist2d(lab_fts_low, prototype_low,
                            scaler=args.scaler) for prototype_low in prototypes_low], dim=1)
            unlab_dist_low = torch.stack([calDist2d(unlab_fts_low, prototype_low,
                            scaler=args.scaler) for prototype_low in prototypes_low], dim=1)


            '''Prototype prediction fusion'''
            proto_l_pred = decoder.ensemble_layer(lab_dist, lab_dist_low)
            proto_u_pred = decoder.ensemble_layer(unlab_dist, unlab_dist_low)

            '''Prototype consistency learning'''
            loss_ce_pt = ce_loss(proto_l_pred, lab_lab.long())
            loss_dice_pt = dice_loss(proto_l_pred, lab_lab)
            loss_focal_pt = focal_loss(proto_l_pred, lab_lab.type(torch.int64))
            loss_iou_pt = iou_loss(proto_l_pred, lab_lab)
            pt_sup_loss = (loss_ce_pt + loss_dice_pt + loss_focal_pt + loss_iou_pt) / 4  # 2
            unsup_loss = consistency_weight * torch.sum((pixel_level_ce_loss(proto_u_pred, pred_u.detach())
                                                                + pixel_level_ce_loss(pred_u_aug, pred_u.detach())
                                                                + pixel_level_ce_loss(pred_u_fp, pred_u.detach())
                                                        ) * reliability_map)

            optimizer_encoder.zero_grad()
            opt_linear_decoder.zero_grad()
            loss = pt_sup_loss + ln_sup_loss + unsup_loss

            loss.backward()

            optimizer_encoder.step()
            opt_linear_decoder.step()

            # torch.cuda.empty_cache()
            logging.info('iters : %d, loss: %.3f, pt_sup_loss: %.3f, ln_sup_loss: %.3f, unsup_loss: %.3f' %
                         (iter_num, loss.item(), pt_sup_loss.item(), ln_sup_loss.item(), unsup_loss.item()))

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

            if iter_num > 300 and iter_num % args.save_step == 0:
                net = infer_net(encoder, decoder)

                metric_list = 0.0
                for _, sampled_batch in enumerate(val_loader):
                    val_image = sampled_batch["image"]
                    val_label = sampled_batch["label"]

                    metric_i = val_2d.test_single_volume(val_image, val_label, net, classes=args.num_classes)
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)
                performance = np.mean(metric_list, axis=0)[0]
                if performance > best_performance:
                    best_performance = performance
                    save_encoder_path = os.path.join(save_path, 'iter_{}_encoder_dice_{}.pth'.format(iter_num, round(
                        best_performance, 4)))
                    save_decoder_path = os.path.join(save_path, 'iter_{}_decoder_dice_{}.pth'.format(iter_num, round(
                        best_performance, 4)))
                    best_encoder_path = os.path.join(save_path, 'best_encoder.pth')
                    best_decoder_path = os.path.join(save_path, 'best_decoder.pth')

                    save_net_opt(encoder, optimizer_encoder, save_encoder_path, iter_num)
                    save_net_opt(decoder, opt_linear_decoder, save_decoder_path, iter_num)
                    save_net_opt(encoder, optimizer_encoder, best_encoder_path, iter_num)
                    save_net_opt(decoder, opt_linear_decoder, best_decoder_path, iter_num)

                    save_net_opt(encoder, optimizer_encoder, save_path / 'encoder.pth', iter_num)
                    save_net_opt(decoder, opt_linear_decoder, save_path / 'decoder.pth', iter_num)

                # TESTACDC(iter_num, phase='self_train')
                logging.info(
                    'iteration %d : mean_dice : %f, val_maxdice : %f' % (iter_num, performance, best_performance))

                File = open(os.path.join(logdir, 'validate.log'), 'a')
                File.write(
                    "iter_num: {}, val_dice: {}, best_dict: {}\n".format(iter_num, performance, best_performance))
                File.close()

        writer.flush()


if __name__ == '__main__':
    # if os.path.exists(resultdir + '/codes'):
    #     shutil.rmtree(resultdir + '/codes')
    # shutil.copytree('.', resultdir + '/codes',
    #                 shutil.ignore_patterns(['.git', '__pycache__']))
    main()
