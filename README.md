# DP-Net
The official implementation of "DP-Net: Dual prototype learning for semi-supervised medical image segmentation"

## Requirements
This repository is based on PyTorch 2.1.1, CUDA 12.1 and Python 3.9.21. 

Install the main packages:
```angular2html
pip install requirements.txt
```
## Usage
We provide `code`, `data_split` and `models` for LA, Pancreas-NIH and ACDC dataset.

Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data)  [Pancreas-NIH](https://github.com/koncle/CoraNet)  and [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC).

To train a model,
```
python train_3d.py --dataset LA --labeled_num 4  #for LA training
python train_3d.py --dataset PAN --labeled_num 6 #for Pancreas-NIH training
python train_2d.py --dataset ACDC --labeled_num 7  #for ACDC training
python train_2d.py --dataset PROMISE --labeled_num 7  #for PROMISE 12 training
``` 

To test a model,
```
python test_3d.py --dataset LA #for LA and testing
python test_3d.py --dataset PAN #for Pancreas-NIH testing
python test_2d.py  #for ACDC and PROMISE 12 testing
```
