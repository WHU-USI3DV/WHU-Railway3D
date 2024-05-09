
![Intro figure](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/doc/Github_intro.png)

Created by Hugues THOMAS

## Introduction

This repository contains the implementation of **Kernel Point Convolution** (KPConv) in [PyTorch](https://pytorch.org/).

KPConv is also available in [Tensorflow](https://github.com/HuguesTHOMAS/KPConv) (original but older implementation).

Another implementation of KPConv is available in [PyTorch-Points-3D](https://github.com/nicolas-chaulet/torch-points3d)


## Installation

This implementation has been tested on Ubuntu 18.04 and Windows 10. Details are provided in [INSTALL.md](./INSTALL.md).

## Experiments

* [Scene Segmentation on Railway3D](./doc/scene_segmentation_guide.md): Instructions to train KP-FCNN on a scene segmentation 
 task (WHU-Railway3D).

### Data

Railway3D dataset can be downloaded <a href="https://forms.gle/HswKqzUWRuG4UQMZ6">here</a>. 
Download the ply files and move it to `/WHU-Railway3D/repos/data`.

N.B. If you want to place your data anywhere else, you just have to change the variable `self.path` of `Railway3DDataset` class. # /KPConv-PyTorch/datasets/Railway3D.py


### Training

First you need to change some paths on /KPConv-PyTorch/datasets/Railway3D.py:
```
self.prefix              # Path where the a specific railway scene dataset is located
self.path                # Path where the WHU-Railway3D dataset is located
self.scene               # Name of the railway scene being processed
self.kp_data_path        # Path where preprocessed data is saved
self.train_file_path_txt # Path to where the train_file_path.txt file is located.
self.val_file_path_txt   # Path to where the val_file_path.txt file is located.
self.test_file_path_txt  # Path to where the test_file_path.txt file is located.
```

Then simply run the following script to start the training:
```python
python3 train_railway3d.py
```
The parameters can be modified in a configuration subclass called `Railway3DConfig`, and the first run of this script might take some time to precompute dataset structures.

Pre-trained models can be found at [Baidu Disk Cloud](https://pan.baidu.com/s/1efEMnVuFHCK2KBex0P9ZzQ?pwd=rail) and [Google Drive](https://drive.google.com/drive/folders/1YKYhIOjmFIkWy_TKSVZ7D0smLnNqfsli?usp=sharing).


### Test the trained model

The test script is the same for all models (segmentation or classification). In `test_any_model.py`, you will find detailed comments explaining how to choose which logged trained model you want to test. Follow them and then run the script :
```python
python3 test_models.py
```


## Acknowledgment

This code uses the <a href="https://github.com/jlblancoc/nanoflann">nanoflann</a> library.

## License
This code is released under MIT License (see LICENSE file for details).

KPConv is a point convolution operator presented in our ICCV2019 paper ([arXiv](https://arxiv.org/abs/1904.08889)). If you find this work useful in your 
research, please consider citing:

```
@article{thomas2019KPConv,
    Author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.},
    Title = {KPConv: Flexible and Deformable Convolution for Point Clouds},
    Journal = {Proceedings of the IEEE International Conference on Computer Vision},
    Year = {2019}
}
```