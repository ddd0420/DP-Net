import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
import os
def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def var_all_case(net, data_root, save_result = True, test_save_path = None, num_classes=2):
    cfg = {
        "LA" : {"patch_size":(112, 112, 80), "stride_xy":18, "stride_z":4},
        "BRA": {"patch_size":(96, 96, 96), "stride_xy":64, "stride_z":64},
        "PAN": {"patch_size":(96, 96, 96), "stride_xy":16, "stride_z":16}
    }
    image_list = []
    if "LA" in data_root:
        stride_xy = cfg['LA']['stride_xy']
        patch_size = cfg['LA']['patch_size']
        stride_z = cfg['LA']['stride_z']
        with open(data_root + '/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = [data_root + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    elif "Pan" in data_root:
        stride_xy = cfg['PAN']['stride_xy']
        patch_size = cfg['PAN']['patch_size']
        stride_z = cfg['PAN']['stride_z']
        with open(data_root + '/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = [data_root + "/Pancreas_h5/" + item.replace('\n', '') + ".h5" for item in image_list]
    elif "Bra" in data_root:
        stride_xy = cfg['BRA']['stride_xy']
        patch_size = cfg['BRA']['patch_size']
        stride_z = cfg['BRA']['stride_z']
        with open(data_root + '/val.txt', 'r') as f:
            image_list = f.readlines()
        image_list = [data_root + "/data/" + item.replace('\n', '') + ".h5" for item in image_list]
    ith = 0
    dc_list = []
    jc_list = []
    hd95_list = []
    asd_list = []
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        single_metric = calculate_metric_percase(prediction, label[:])
        # total_metric += np.asarray(single_metric)
        # test_set_metrics.append([single_metric[0], single_metric[1], single_metric[2], single_metric[3]])
        dc_list.append(single_metric[0])
        jc_list.append(single_metric[1])
        hd95_list.append(single_metric[2])
        asd_list.append(single_metric[3])

        if save_result:
            visual_path = os.path.join(test_save_path, "../visualization")
            if not os.path.exists(visual_path):
                os.makedirs(visual_path)
            # visual_path1 = os.path.join(visual_path, "%02d_pred.nii.gz" % ith)
            # visual_path2 = os.path.join(visual_path, "%02d_scores_pred.nii.gz" % ith)
            # visual_path3 = os.path.join(visual_path, "%02d_gt.nii.gz" % ith)

            # nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), visual_path1)
            # nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), visual_path2)
            # nib.save(nib.Nifti1Image(score_map[:].astype(np.float32), np.eye(4)), visual_path3)
            
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), visual_path +"/"+   "%02d_pred.nii.gz" % ith)
            #nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), visual_path +"/"+  "%02d_img.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), visual_path +"/"+  "%02d_gt.nii.gz" % ith)

            ith += 1
        # if np.sum(prediction)==0:
        #     dice = 0
        # else:
        #     dice = metric.binary.dc(prediction, label)
    dc_arr = np.array(dc_list)
    jc_arr = np.array(jc_list)
    hd95_arr = np.array(hd95_list)
    asd_arr = np.array(asd_list)

    dice_mean = np.mean(dc_arr)
    dice_std = np.std(dc_arr)

    jc_mean = np.mean(jc_arr)
    jc_std = np.std(jc_arr)

    hd95_mean = np.mean(hd95_arr)
    hd95_std = np.std(hd95_arr)

    assd_mean = np.mean(asd_arr)
    assd_std = np.std(asd_arr)

    avg_metric = [dice_mean, jc_mean, hd95_mean, assd_mean]
    std_metric = [dice_std, jc_std, hd95_std, assd_std]
    print("Validation end")
    return avg_metric, std_metric

def test_all_case(model, image_list, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True, test_save_path=None, preproc_fn=None, metric_detail=0, nms=0):
    
    loader = tqdm(image_list) if not metric_detail else image_list
    total_metric = 0.0
    ith = 0
    test_set_metrics = []
    for image_path in loader:
        # id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if nms:
            prediction = getLargestCC(prediction)
            
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
            
        if metric_detail:
            print('%02d,\t%.5f, %.5f, %.5f, %.5f' % (ith, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))

        total_metric += np.asarray(single_metric)
        test_set_metrics.append([single_metric[0], single_metric[1], single_metric[2], single_metric[3]])
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path +  "%02d_pred.nii.gz" % ith)
            nib.save(nib.Nifti1Image(score_map[0].astype(np.float32), np.eye(4)), test_save_path +  "%02d_scores.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + "%02d_gt.nii.gz" % ith)

        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    std_var = np.std(test_set_metrics, axis=0)

    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))
        f.writelines('standard metric is {} \n'.format(std_var))
    return avg_metric


def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1 = model(test_patch)
                    if len(y1) > 1:
                        y1 = y1[0]
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(np.int64)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice * 100, jc * 100, hd, asd
