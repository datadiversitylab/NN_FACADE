# Neural Network-based extraction of Building Facade Color patterns from Google Street View Images

In this project, I focus on extracting building facades from street view imagery using the University of Central Florida's Google Street View dataset, which includes 62,058 images from Pittsburgh, Orlando, and parts of Manhattan. To address variations in environmental conditions like weather and lighting, I implement a [Deep White-Balance Editing](https://openaccess.thecvf.com/content_CVPR_2020/html/Afifi_Deep_White-Balance_Editing_CVPR_2020_paper.html) approach, by Afifi and Brown. For the main task of facade extraction, I utilize the [kMaX-DeepLab model](https://arxiv.org/abs/2207.04044) by Yu et al., which is designed for precise image segmentation tasks.

# Getting Started

This section outlines the steps to preprocess images and perform image segmentation using the provided scripts.

### Preprocessing Image White Balance Correction
To perform white balance correction, navigate to the `AWB_preprocessed_model` folder and run the script:
```bash
python AWB_preprocessed.py
```
### Image segmentation
To perform image segmentation run the script:
```bash
python facades.py
```
# Setup your Conda environment to run code locally
### Preprocession
Create your conda environment with the name you select
```bash
conda create -n name_of_your_environment python=3.12
```
Install the required pacakage to your environment There is a [white_balance_env.yml](https://github.com/datadiversitylab/NN_FACADE/blob/main/white_balance_env.yml) file you can use to replicate the conda environment!
### Image segmentation
Create your conda environment with the name you select
```bash
conda create -n name_of_your_environment python=3.11
```
Install the required pacakage to your environment There is a [building_facades_env.yml](https://github.com/datadiversitylab/NN_FACADE/blob/main/building_facades_env.yml) file you can use to replicate the conda environment!

# Citation
### Dataset
```bash
@article{zamir2014image,
  title={Image Geo-localization Based on Multiple Nearest Neighbor Feature Matching using Generalized Graphs},
  author={Zamir, Amir Roshan and Shah, Mubarak},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2014}
}
```
### Preprocession
```bash
@inproceedings{afifi2020deepWB,
  title={Deep White-Balance Editing},
  author={Afifi, Mahmoud and Brown, Michael S},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
### Image segmentation
```bash
@article{deeplab2_2021,
  author={Mark Weber and Huiyu Wang and Siyuan Qiao and Jun Xie and Maxwell D. Collins and Yukun Zhu and Liangzhe Yuan and Dahun Kim and Qihang Yu and Daniel Cremers and Laura Leal-Taixe and Alan L. Yuille and Florian Schroff and Hartwig Adam and Liang-Chieh Chen},
  title={{DeepLab2: A TensorFlow Library for Deep Labeling}},
  journal={arXiv: 2106.09748},
  year={2021}
}
```
