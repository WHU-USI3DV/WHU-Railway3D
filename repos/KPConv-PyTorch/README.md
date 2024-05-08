
![Intro figure](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/doc/Github_intro.png)

Created by Hugues THOMAS

## Introduction

This repository contains the implementation of **Kernel Point Convolution** (KPConv) in [PyTorch](https://pytorch.org/).

KPConv is also available in [Tensorflow](https://github.com/HuguesTHOMAS/KPConv) (original but older implementation).

Another implementation of KPConv is available in [PyTorch-Points-3D](https://github.com/nicolas-chaulet/torch-points3d)

KPConv is a point convolution operator presented in our ICCV2019 paper ([arXiv](https://arxiv.org/abs/1904.08889)). If you find our work useful in your 
research, please consider citing:

```
@article{thomas2019KPConv,
    Author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.},
    Title = {KPConv: Flexible and Deformable Convolution for Point Clouds},
    Journal = {Proceedings of the IEEE International Conference on Computer Vision},
    Year = {2019}
}
```

## Installation

This implementation has been tested on Ubuntu 18.04 and Windows 10. Details are provided in [INSTALL.md](./INSTALL.md).

## Experiments

We provide scripts for three experiments: ModelNet40, S3DIS and SemanticKitti. The instructions to run these 
experiments are in the [doc](./doc) folder.

* [Scene Segmentation](./doc/scene_segmentation_guide.md): Instructions to train KP-FCNN on a scene segmentation 
 task (WHU-Railway3D).

## Acknowledgment

Our code uses the <a href="https://github.com/jlblancoc/nanoflann">nanoflann</a> library.

## License
Our code is released under MIT License (see LICENSE file for details).

