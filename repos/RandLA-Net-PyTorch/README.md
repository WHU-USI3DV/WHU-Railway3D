# RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

This repository contains a PyTorch implementation of [RandLA-Net](http://arxiv.org/abs/1911.11236) on Railway3D.

**This repository is mainly based on the [repository](https://github.com/qiqihaer/RandLA-Net-pytorch)**.

## Preparation(Railway3D as example)

1. Clone this repository.
2. Install some Python dependencies, such as scikit-learn. All packages can be installed with pip.
3. Environment:
   ```
   ubuntu 18.04
   python 3.7.16
   torch 1.12.1
   numpy 1.21.5
   torchvision 0.13.1
   scikit-learn 0.22.2
   pandas 1.3.5
   tqdm 4.64.1
   Cython 0.29.33 (Cython is important!)
   ```

4. Install python functions. the functions and the codes are copied from the [official implementation with Tensorflow](https://github.com/QingyongHu/RandLA-Net).
   ```
   sh compile_op.sh
   ```
   > Attention: please check out *./utils/nearest_neighbors/lib/python/KNN_NanoFLANN-0.0.0-py3.7-linux-x86_64.egg/* and copy the **.so** file to the parent folder.
   > 
   > **Update in 2023.2.23: We provide a **.so** file for python3.7, and you don't need to compile the cpp code if you are using python3.7.**

5. Download the WHU-Railway3D[dataset](https://forms.gle/HswKqzUWRuG4UQMZ6), and preprocess the data:
   ```python
   python utils/data_prepare_railway3d.py
   ```

   Note: Please change the dataset path in the 'data_prepare_railway3d.py' with your own path.

## Train a model(Railway3D as example)

First you need to change some paths on Railway3D_dataset.py:
   ```python
   self.prefix              # Path where the a specific railway scene dataset is located
   self.path                # Path where preprocessed data is saved
   self.train_file_path_txt # Path to where the train_file_path.txt file is located.
   self.val_file_path_txt   # Path to where the val_file_path.txt file is located.
   self.test_file_path_txt  # Path to where the test_file_path.txt file is located.
   ```

Then simply run the following script to start the training:
   ```python
   python main_Railway3D.py
   ```

## Test a model(Railway3D as example)

   ```python
   python test_Railway3D.py
   ```

## Results

### Railway3D

We train this network for 100 epoches, and the eval results(after voting) in the test set are as follows: mIoU = 70.5%

```
eval accuracy: 91.0%
Mean IoU = 70.5%
IoU of  class_rails = 66.4
IoU of  class_track bed = 88.3
IoU of  class_masts = 70.9
IoU of  class_support devices = 45.1
IoU of  class_overhead lines = 79.0
IoU of  class_fences = 57.2
IoU of  class_poles = 46.3
IoU of  class_vegetation = 85.9
IoU of  class_buildings = 69.4
IoU of  class_ground = 88.6
IoU of  class_other = 78.4
--------------------------------------------------------------------------
 70.5 |  66.4  88.3  70.9  45.1  79.0  57.2  46.3  85.9  69.4  88.6  78.4 
--------------------------------------------------------------------------
```

The checkpoint is in the output folder.

### Model
Pre-trained models can be found at [Baidu Disk Cloud](https://pan.baidu.com/s/1efEMnVuFHCK2KBex0P9ZzQ?pwd=rail) and [Google Drive](https://drive.google.com/drive/folders/1YKYhIOjmFIkWy_TKSVZ7D0smLnNqfsli?usp=sharing).


