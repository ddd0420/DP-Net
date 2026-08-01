# DP-Net
The official implementation of "DP-Net: Dual prototype learning for semi-supervised medical image segmentation"

## Requirements
This repository is based on PyTorch 2.1.1, CUDA 12.1 and Python 3.9.21. All experiments in our paper were conducted on an NVIDIA RTX 4090 GPU with an identical experimental setting.

Install the main packages:
```angular2html
pip install requirements.txt
```
## Usage
We provide `code`, `data_split` and `models` for LA, Pancreas-NIH and ACDC dataset.

Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data),  [Pancreas-NIH](https://github.com/koncle/CoraNet), 
and [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC), [PROMISE 12]([https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC](https://github.com/wxfaaaaa/DCNet))

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

## Acknowledgements
Part of the code is adapted from [UPCoL](https://github.com/ycwu1997/MC-Net), [SSNet](https://github.com/ycwu1997/SS-Net), and [SSL4MIS](https://github.com/HiLab-git/SSL4MIS). Thanks to these authors for their valuable works and hope our model can promote the relevant research as well.
## Questions
If you have any questions, welcome contact me at 'dgx@whut.edu.cn'
