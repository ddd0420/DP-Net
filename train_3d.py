from pathlib import Path
from tqdm import tqdm
import sys
import argparse
import shutil

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
from networks.VNet import VNet
from skimage.measure import label

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
    parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
    parser.add_argument('--alpha', type=float, default=0.99, help='params in ema update')
    parser.add_argument('--resume', action='store_true')
    # dataset
    parser.add_argument('--labeled_num', type=int, default=8, help='set the seed of random initialization')
    parser.add_argument('--dataset', type=str, default="LA")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/dataset/LA", help="Path to the dataset.")
    parser.add_argument("--save_path", type=str, default='./results', help="Path to save.")

    # Optimization options
    parser.add_argument('--lab_batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--unlab_batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate 0.001')
    parser.add_argument('--iters', type=int, default=15000, help='maximum iter number for training')
    parser.add_argument('--save_step', type=int, default=500, help='frequecy of checkpoint save in pretraining')
    parser.add_argument('--consistency_rampup', type=float, default=10000.0, help='consistency_rampup')

    parser.add_argument('--p_weight', type=float, default=0.5, help='weight of prototype loss')
    parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unsupervised loss')

    parser.add_argument('--beta1', type=float, default=0.5, help='params of optimizer Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='params of optimizer Adam')
    parser.add_argument('--scaler', type=float, default=1, help='multiplier of prototype')
    # Miscs
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--seed', type=int, default=1337, help='set the seed of random initialization')
    return parser.parse_args()


args = get_arguments()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# create logger
resultdir = "{}/{}_pu{}{}_{}_labeled/".format(args.save_path, args.dataset, args.p_weight, args.u_weight, args.labeled_num)
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

def create_dataloader(label_num=8):
    valset = None
    max_samples = 0
    if 'LA' in data_dir:
        max_samples = 80
        train_labset = LAHeart(data_dir)
    elif 'Bra' in data_dir:
        max_samples = 250
        train_labset = BraTS2019(data_dir)
    elif "Pan" in data_dir:
        max_samples = 62
        train_labset = Pancreas(data_dir)

    labelnum = args.labeled_num
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs,
                                          args.lab_batch_size + args.unlab_batch_size,
                                          args.unlab_batch_size)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(train_labset, batch_sampler=batch_sampler, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    logging.info("{} iterations per epoch.".format(len(trainloader)))
    return trainloader


def proto_align_loss(proto1, proto2):
    loss = 0.0
    for c in range(args.num_classes):
        dot_product = torch.dot(proto1[c].view(-1), proto2[c].view(-1))
        norm_tensor1 = torch.norm(proto1[c])
        norm_tensor2 = torch.norm(proto2[c])
        loss = loss + 1 - (dot_product / (norm_tensor1 * norm_tensor2))
    return loss

def create_model(ema=False):
    # net = nn.DataParallel(VNet_AMC(n_channels=1, n_classes=args.num_classes, n_branches=4))
    net = VNet(n_channels=1, n_classes=args.num_classes, normalization='batchnorm').cuda()
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

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

    model = create_model().cuda()
    ema_model = create_model(ema=True).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    trainloader = create_dataloader(args.labeled_num)
    dice_loss = DiceLoss(nclass=args.num_classes)
    ce_loss = CrossEntropyLoss()
    focal_loss = FocalLoss()
    iou_loss = SoftIoULoss(nclass=args.num_classes)
    pixel_level_ce_loss = CrossEntropyLoss(reduction='none')

    best_dice = 0.0
    iter_num = 0
    from torchvision import transforms
    if args.dataset == "Bra":
        args.scaler = 1.0
        color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    else:
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    random_jittor = transforms.RandomApply([color_jitter], p=1.0)

    if "LA" in data_dir:
        patch_size = (112, 112, 80)
    else:
        patch_size = (96, 96, 96)

    epoch_num = 1 + args.iters / len(trainloader)
    for _ in tqdm(range(int(epoch_num)), ncols=70):
        logging.info('\n')
        model.train()
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            iter_num += 1
            lab_img, lab_lab = volume_batch[:args.lab_batch_size].cuda(), label_batch[:args.lab_batch_size].cuda()
            unlab_img = volume_batch[args.lab_batch_size:].cuda()
            lab_lab_onehot = to_one_hot(lab_lab.unsqueeze(1), args.num_classes)

            reshaped_image = unlab_img.clone().view(args.unlab_batch_size * patch_size[0], 1, patch_size[1], patch_size[2])
            aug_unlab = random_jittor(reshaped_image).view(args.unlab_batch_size, 1, patch_size[0], patch_size[1], patch_size[2])

            '''Supervised'''
            lab_out, feature_lab = model(lab_img)
            aug_unlab_out, fp_unlab_out, feature_unlab_s = model(aug_unlab, fea_p=True)
            unlab_aug_soft = torch.softmax(aug_unlab_out, dim=1)
            unlab_fp_soft = torch.softmax(fp_unlab_out, dim=1)

            loss_ce = ce_loss(lab_out, lab_lab)
            loss_dice = dice_loss(lab_out, lab_lab)
            loss_focal = focal_loss(lab_out, lab_lab)
            loss_iou = iou_loss(lab_out, lab_lab)
            sup_loss = (loss_ce + loss_dice + loss_focal + loss_iou) / 4

            # labeled prototypes
            lab_fts = F.interpolate(feature_lab[4], size=lab_lab.shape[-3:],mode='trilinear')
            lab_prototypes = getPrototype(lab_fts, lab_lab_onehot)

            lab_fts1 = F.interpolate(feature_lab[-4], size=lab_lab.shape[-3:], mode='trilinear')
            lab_prototypes1 = getPrototype(lab_fts1, lab_lab_onehot)

            '''Unsupervised'''
            with torch.no_grad():
                unlab_out, feature_unlab_t = ema_model(unlab_img)
                unlab_soft = torch.softmax(unlab_out, dim=1)
                unlab_mask = torch.argmax(unlab_soft, dim=1)
                unlab_onehot = to_one_hot(unlab_mask.unsqueeze(1), args.num_classes)

                # uncertainty assesment
                uncertainty =  -torch.sum(unlab_soft * torch.log(unlab_soft  + 1e-16), dim=1)
                norm_uncertainty = torch.stack([uncertain / torch.sum(uncertain) for uncertain in uncertainty],dim=0)

                reliability_map = (1 - norm_uncertainty) / np.prod(np.array(norm_uncertainty.shape[-3:]))

                # unlabeled prototypes
                unlab_fts = F.interpolate(feature_unlab_t[4], size=lab_lab.shape[-3:],mode='trilinear')
                unlab_prototypes = getPrototype(unlab_fts, unlab_onehot, reliability_map)
                unlab_fts1 = F.interpolate(feature_unlab_t[-4], size=lab_lab.shape[-3:], mode='trilinear')
                unlab_prototypes1 = getPrototype(unlab_fts1, unlab_onehot, reliability_map)

                consistency_weight = get_current_consistency_weight(iter_num, args.consistency_rampup)

            '''Prototype fusion'''
            prototypes = [(lab_prototypes[c] + consistency_weight * unlab_prototypes[c]) / (1 + consistency_weight)
                          for c in range(args.num_classes)]
            lab_dist = torch.stack([calDist(lab_fts, prototype, scaler=args.scaler) for prototype in prototypes], dim=1)
            unlab_dist = torch.stack([calDist(unlab_fts, prototype, scaler=args.scaler) for prototype in prototypes], dim=1)

            prototypes1 = [(lab_prototypes1[c] + consistency_weight * unlab_prototypes1[c]) / (1 + consistency_weight)
                           for c in range(args.num_classes)]
            lab_dist1 = torch.stack([calDist(lab_fts1, prototype1, scaler=args.scaler) for prototype1 in prototypes1], dim=1)
            unlab_dist1 = torch.stack([calDist(unlab_fts1, prototype1, scaler=args.scaler) for prototype1 in prototypes1], dim=1)
            # print("lab_dist.shape",lab_dist[-1].shape)
            # print("lab_dist1.shape",lab_dist1[-1].shape)

            proto_l_pred = model.ensemble_layer(lab_dist, lab_dist1)
            proto_u_pred = model.ensemble_layer(unlab_dist, unlab_dist1)

            '''Prototype consistency learning'''
            loss_ce2 = ce_loss(proto_l_pred, lab_lab)
            loss_dice2 = dice_loss(proto_l_pred, lab_lab)
            loss_focal2 = focal_loss(proto_l_pred, lab_lab)
            loss_iou2 = iou_loss(proto_l_pred, lab_lab)
            proto_sup_loss = (loss_ce2 + loss_dice2 + loss_focal2 + loss_iou2) / 4  # 2
            unsup_loss = consistency_weight * torch.sum((pixel_level_ce_loss(proto_u_pred, unlab_soft.detach())
                                                        + pixel_level_ce_loss(unlab_aug_soft, unlab_soft.detach())
                                                        + pixel_level_ce_loss(unlab_fp_soft, unlab_soft.detach())
                                                        ) * reliability_map)
            loss = sup_loss + args.p_weight * proto_sup_loss + unsup_loss * args.u_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.alpha, iter_num)

            torch.cuda.empty_cache()

            logging.info(
                'iters : %d, loss: %.4f, sup_loss: %.4f, proto_sup_loss: %.4f, unsup_loss: %.4f' %
                (iter_num, loss.item(), sup_loss.item(),proto_sup_loss.item(), unsup_loss.item()))

            if  iter_num>3000 and iter_num % args.save_step == 0:
                dice_sample, best_dice, max_flag = test_all(model, data_dir, best_dice, test_save_path=save_path)

                File = open(os.path.join(logdir, 'validate.log'), 'a')
                File.write("iter_num: {}, val_dice: {}, best_dict: {}\n".format(iter_num, dice_sample, best_dice))
                File.close()

                if max_flag:
                # if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    save_mode_path = os.path.join(save_path,'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(save_path, 'VNet_best_model.pth')

                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                    logging.info("save best model to {}".format(save_mode_path))

                logging.info('iteration %d :mean_dice: %f best_dice: %f' % (iter_num, dice_sample, best_dice))

                model.train()

            if iter_num >= args.iters:
                break
        writer.flush()


def test_all(net, data_root, maxdice=0.0, test_save_path=None):
    avg_metrics, std_metrics = test_3d_patch.var_all_case(net, data_root, test_save_path=test_save_path)
    val_dice = avg_metrics[0]
    if val_dice > maxdice:
        maxdice = val_dice
        max_flag = True
    else:
        max_flag = False

    logging.info('Evaluation : val_dice: %.4f, val_maxdice: %.4f\n' % (val_dice, maxdice))

    logging.info('\nDice:')
    logging.info('Mean :%.2f(%.2f)' % (avg_metrics[0], std_metrics[0]))

    logging.info('\nJaccard:')
    logging.info('Mean :%.2f(%.2f)' % (avg_metrics[1], std_metrics[1]))

    logging.info('\nHD95:')
    logging.info('Mean :%.2f(%.2f)' % (avg_metrics[2], std_metrics[2]))

    logging.info('\nASSD:')
    logging.info('Mean :%.2f(%.2f)' % (avg_metrics[3], std_metrics[3]))
    if "Bra" in data_root:
        if max_flag:
            avg_metrics, std_metrics = test_3d_patch.var_all_case(net, data_root, test_save_path=test_save_path)
            val_dice = avg_metrics[0]
            if val_dice > maxdice:
                maxdice = val_dice
                max_flag = True
            else:
                max_flag = False

            logging.info('Evaluation : test_dice: %.4f, test_maxdice: %.4f\n' % (val_dice, maxdice))

            logging.info('\nDice:')
            logging.info('Mean :%.2f(%.2f)' % (avg_metrics[0], std_metrics[0]))

            logging.info('\nJaccard:')
            logging.info('Mean :%.2f(%.2f)' % (avg_metrics[1], std_metrics[1]))

            logging.info('\nHD95:')
            logging.info('Mean :%.2f(%.2f)' % (avg_metrics[2], std_metrics[2]))

            logging.info('\nASSD:')
            logging.info('Mean :%.2f(%.2f)' % (avg_metrics[3], std_metrics[3]))
            return val_dice, maxdice, max_flag
        return 0.0, 0.0, False
    return val_dice, maxdice, max_flag

if __name__ == '__main__':

    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    shutil.copy(sys.argv[0], resultdir + '/')

    main()
