# imTED: Integrally Migrating Pre-trained Transformer Encoder-decoders for Visual Object Detection

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/integral-migrating-pre-trained-transformer/few-shot-object-detection-on-ms-coco-30-shot)](https://paperswithcode.com/sota/few-shot-object-detection-on-ms-coco-30-shot?p=integral-migrating-pre-trained-transformer)
<a href="https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Integrally_Migrating_Pre-trained_Transformer_Encoder-decoders_for_Visual_Object_Detection_ICCV_2023_paper.html"><img src="https://img.shields.io/badge/ICCV2023-Paper-<color>"></a>
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2205.09613)

<!-- <div align=center><img src="figs/Framework1.png"></div> -->
<div align=center><img src="figs/Framework2.png"></div>
<!-- <div align=center><img src="figs/Framework3.png"></div> -->

Code of our ICCV 2023 paper [Integrally Migrating Pre-trained Transformer Encoder-decoders for  Visual Object Detection](https://arxiv.org/abs/2205.09613). Blog in Chinese is available [here](https://zhuanlan.zhihu.com/p/645282546).

The code is based on [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v2.11.0), please refer to [get_started.md](docs/get_started.md) and [MMDET_README.md](MMDET_README.md) to set up the environment and prepare the data.

## Config Files and Performance and Trained Weights

We provide 9 configuration files in the configs directory.

| Config File                                                                        | Backbone    | Epochs    | Box AP      | Mask AP   | Download |
| :--------------------------------------------------------------------------------: | :---------: | :-------: | :---------: | :-------: | :-------: |
| [imted_faster_rcnn_vit_small_3x_coco](configs/imted/imted_faster_rcnn_vit_small_3x_coco.py)                               | ViT-S       | 36        | 48.2        |           | [model](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/liufeng20_mails_ucas_ac_cn/EQaRZ_hrWolAr0BqGhv3PzgB6T9s-HHKIxwJvoXeDFHrrQ?e=I3Dul5) |
| [imted_faster_rcnn_vit_base_3x_coco](configs/imted/imted_faster_rcnn_vit_base_3x_coco.py)                                | ViT-B       | 36        | 52.9        |           | [model](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/liufeng20_mails_ucas_ac_cn/EX11K6R1X7VGilexd6WEapoBQYLa2ZofYGXlyVLB8TJlFA?e=8gSCTg) |
| [imted_faster_rcnn_vit_large_3x_coco](configs/imted/imted_faster_rcnn_vit_large_3x_coco.py)                               | ViT-L       | 36        | 55.4        |           | [model](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/liufeng20_mails_ucas_ac_cn/EW-QTq_TxxNFtMJBIn7Tfr0BHG6RXp-Yob7NirlZEzcX1A?e=atschp) |
| [imted_mask_rcnn_vit_small_3x_coco](configs/imted/imted_mask_rcnn_vit_small_3x_coco.py)                                 | ViT-S       | 36        | 48.7        | 42.7      | [model](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/liufeng20_mails_ucas_ac_cn/EbowkBw7LkJCjac4Ptza6HwB5VoKna-CJci8pezNNcRggA?e=GklD74) |
| [imted_mask_rcnn_vit_base_3x_coco](configs/imted/imted_mask_rcnn_vit_base_3x_coco.py)                                  | ViT-B       | 36        | 53.3        | 46.4      | [model](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/liufeng20_mails_ucas_ac_cn/EdXyeZpXRKtMurx-m-gzLSIBgqibthpJumIvLjni8MUNWw?e=dB454E) |
| [imted_mask_rcnn_vit_large_3x_coco](configs/imted/imted_mask_rcnn_vit_large_3x_coco.py)                                 | ViT-L       | 36        | 55.5        | 48.1      | [model](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/liufeng20_mails_ucas_ac_cn/EZu-46TJEjFKvy3mHbE4FlkBwSHGntKlMeDxanXfCoIJAA?e=33KZO0) |
| [imted_faster_rcnn_vit_base_2x_base_training_coco](configs/imted/few_shot/imted_faster_rcnn_vit_base_2x_base_training_coco.py)         | ViT-B       | 24        | 50.6        |           | [model](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/liufeng20_mails_ucas_ac_cn/EWedyWJx8S5Hi-8S0TGBxZwBBd7mxSFD0rvdiYqdcWXSxA?e=gra0ao) |
| [imted_faster_rcnn_vit_base_2x_finetuning_10shot_coco](configs/imted/few_shot/imted_faster_rcnn_vit_base_2x_finetuning_10shot_coco.py)     | ViT-B       | 108       | 23.0        |           | [model](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/liufeng20_mails_ucas_ac_cn/ETGkGkfywcJCuR6FzaMs21YBuHQ_7jyCYKVfj4kG46cuAQ?e=1YHKce) |
| [imted_faster_rcnn_vit_base_2x_finetuning_30shot_coco](configs/imted/few_shot/imted_faster_rcnn_vit_base_2x_finetuning_30shot_coco.py)     | ViT-B       | 108       | 30.4        |           | [model](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/liufeng20_mails_ucas_ac_cn/EYK3tlqxWC9OiPrYi6TFycsBN-AfvbN8YIEskmpat8MZdA?e=WfnrXB) |

## MAE Pre-training

The pre-trained model is trained with the [official MAE code](https://github.com/facebookresearch/mae). 
For ViT-S, we use a 4-layer decoder with dimension 256 for 800 epochs of pre-training. 
For ViT-B, we use an 8-layer decoder with dimension 512 for 1600 epochs of pre-training. Pre-trained weights can be downloaded from the [official MAE weight](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth).
For ViT-L, we use an 8-layer decoder with dimension 512 for 1600 epochs of pre-training. Pre-trained weights can be downloaded from the [official MAE weight](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large_full.pth).

## Last Step of Preparation
For all experiments, remember to modify the path of pre-trained weights in the configuration files, e.g. configs/imted/imted_faster_rcnn_vit_small_3x_coco.py.

For few-shot experiments, please refer to [FsDet](https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/README.md#:~:text=2%2C%20and%203.-,COCO%3A,-cocosplit/%0A%20%20datasplit/%0A%20%20%20%20trainvalno5k) for data preparation. Remember to modify the path of json in the configuration files, e.g. configs/imted/few_shot/imted_faster_rcnn_vit_base_2x_base_training_coco.py. Json files used for few-shot training and evaluation can also be downloaded from [here](https://mailsucasaccn-my.sharepoint.com/:u:/g/personal/liufeng20_mails_ucas_ac_cn/EdDDuaUiWvlImzOHddcsGfkBHp0Me1zFMm20zGHz9l8alg?e=U3gx2I).

## Evaluating with 1 GPU

```bash
tools/dist_test.sh "path/to/config/file.py" "path/to/trained/weights.pth" 1 --eval bbox
```

## Training with 8 GPUs

```bash
tools/dist_train.sh "path/to/config/file.py" 8 
```
## Few-shot Training with 8 GPUs

### Base Training
```bash
tools/dist_train.sh configs/imted/few_shot/imted_faster_rcnn_vit_base_2x_base_training_coco.py 8 
```

### Finetuning
Replace the the ckeckpoint path of your own checkpoint from base training or just use our provided checkpoint [here](https://github.com/LiewFeng/imTED/blob/main/configs/imted/few_shot/imted_faster_rcnn_vit_base_2x_finetuning_30shot_coco.py#L6C1-L6C14). 
```bash
tools/dist_train.sh configs/imted/few_shot/imted_faster_rcnn_vit_base_2x_finetuning_30shot_coco.py 8 
```

## Acknowledgement
This project is based on [MAE](https://github.com/facebookresearch/mae), [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v2.11.0) and [timm](https://github.com/huggingface/pytorch-image-models). Thanks for their wonderful works.

## Some works based on imTED
* [Spatial Transform Decoupling for **Oriented Object Detection**](https://github.com/yuhongtian17/Spatial-Transform-Decoupling)
* [AttentionShift: Iteratively Estimated Part-Based Attention Map for **Pointly Supervised Instance Segmentation**](https://openaccess.thecvf.com/content/CVPR2023/html/Liao_AttentionShift_Iteratively_Estimated_Part-Based_Attention_Map_for_Pointly_Supervised_Instance_CVPR_2023_paper.html)
* [Proposal Distribution Calibration for **Few-Shot Object Detection**](https://github.com/Bohao-Lee/PDC)


## Citation

If you find imTED is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.
```bibtex
@inproceedings{liu2023integrally,
  title={Integrally Migrating Pre-trained Transformer Encoder-decoders for Visual Object Detection},
  author={Liu, Feng and Zhang, Xiaosong and Peng, Zhiliang and Guo, Zonghao and Wan, Fang and Ji, Xiangyang and Ye, Qixiang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6825--6834},
  year={2023}
}
```
