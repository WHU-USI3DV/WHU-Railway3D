
## Scene Segmentation on Railway3D

### Data

We consider our experiment folder is located at `XXXX/Experiments/KPConv-PyTorch`. And we use a common Data folder 
loacated at `XXXX/Data`. Therefore the relative path to the Data folder is `../../Data`.

Railway3D dataset can be downloaded <a href="https://forms.gle/HswKqzUWRuG4UQMZ6">here</a>. 
Download the ply files and move it to `/WHU-Railway3D/repos/data`.

N.B. If you want to place your data anywhere else, you just have to change the variable `self.path` of `Railway3DDataset` class. # /WHU-Railway3D/repos/KPConv-PyTorch/datasets/Railway3D.py


### Training

First you need to change some paths on /datasets/Railway3D.py:
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