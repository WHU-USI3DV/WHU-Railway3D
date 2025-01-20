## 📌 Hyperparameters settings

Indeed, network hyperparameters can have a significant impact on network performance. In fact, after conducting various trials, we have selected appropriate hyperparameters for each baseline. With these hyperparameter settings, different networks were able to learn effectively and achieve satisfactory semantic segmentation results. While our parameter settings may not be optimal, fine-tuning the hyperparameters for different networks is not the focus of this study. Additionally, variations in computer hardware devices can also affect experimental results. Therefore, we leave these tasks to future researchers. They can refer to our parameter settings and further fine-tune them based on their hardware devices to achieve better performance for different networks.

### 1.1 KPConv

[KPconv: Flexible and Deformable Convolution for Point Clouds](https://github.com/HuguesTHOMAS/KPConv)

```python
# Number of kernel points
num_kernel_points = 15
# Radius of the input sphere (decrease value to reduce memory cost)
in_radius = 3.0
# Size of the first subsampling grid in meter (increase value to reduce memory cost)
first_subsampling_dl = 0.06
# Radius of convolution in "number grid cell". (2.5 is the standard value)
conv_radius = 2.5
# Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
deform_radius = 5.0
# Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
KP_extent = 1.2
# Behavior of convolutions in ('constant', 'linear', 'gaussian')
KP_influence = 'linear'
# Aggregation function of KPConv in ('closest', 'sum')
aggregation_mode = 'sum'
# Choice of input features
first_features_dim = 128
in_features_dim = 5 # need to be considered
# Can the network learn modulations
modulated = False
# Batch normalization parameters
use_batch_norm = True
batch_norm_momentum = 0.02
# Deformable offset loss
# 'point2point' fitting geometry by penalizing distance from deform point to input points
# 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
deform_fitting_mode = 'point2point'
deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

#####################
# Training parameters
#####################

# Maximal number of epochs
max_epoch = 200 # need to be considered
# Learning rate management
learning_rate = 1e-2
momentum = 0.98
lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
grad_clip_norm = 100.0
# Number of batch (decrease to reduce memory cost, but it should remain > 3 for stability)
batch_num = 6
# Number of steps per epochs
epoch_steps = 500 # need to be considered
# Number of validation examples per epoch
validation_size = 50 # need to be considered
# Number of epoch between each checkpoint
checkpoint_gap = 50
# Augmentations
augment_scale_anisotropic = True
augment_symmetries = [True, False, False]
augment_rotation = 'vertical'
augment_scale_min = 0.9
augment_scale_max = 1.1
augment_noise = 0.001
augment_color = 0.8
```



### 1.2 RandLA-Net

[RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://github.com/QingyongHu/RandLA-Net)

```python
k_n = 16  # KNN
num_layers = 5  # Number of layers
num_points = 40960  # Number of input points
num_classes = 11  # Number of valid classes
sub_grid_size = 0.06  # preprocess_parameter
batch_size = 6  # batch_size during training
val_batch_size = 20  # batch_size during validation and test
train_steps = 500  # Number of steps per epochs
val_steps = 100  # Number of validation steps per epoch
sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
d_out = [16, 64, 128, 256, 512]  # feature dimension
noise_init = 3.5  # noise initial parameter
max_epoch = 100  # maximum epoch during training
learning_rate = 1e-2  # initial learning rate
lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate
train_sum_dir = 'train_log'
saving = True
saving_path = None
```



### 1.3 SCF-Net

[SCF-Net: Learning Spatial Contextual Features for Large-Scale Point Cloud Segmentation](https://github.com/leofansq/SCF-Net)

```python
k_n = 16  # KNN
num_layers = 5  # Number of layers
num_points = 65536  # Number of input points
num_classes = 11  # Number of valid classes
sub_grid_size = 0.06  # preprocess_parameter
batch_size = 1  # batch_size during training
val_batch_size = 1  # batch_size during validation and test
train_steps = 500  # Number of steps per epochs
val_steps = 100  # Number of validation steps per epoch

sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
d_out = [16, 64, 128, 256, 512]  # feature dimension

noise_init = 3.5  # noise initial parameter
# max_epoch need to reset
max_epoch = 100  # maximum epoch during training
learning_rate = 1e-2  # initial learning rate
lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

train_sum_dir = 'train_log'
saving = True
saving_path = None

augment_scale_anisotropic = True
augment_symmetries = [True, False, False]
augment_rotation = 'vertical'
augment_scale_min = 0.8
augment_scale_max = 1.2
augment_noise = 0.001
augment_occlusion = 'none'
augment_color = 0.8
```



### 1.4 BAAF-Net

[BAAF-Net: Semantic Segmentation for Real Point Cloud Scenes via Bilateral Augmentation and Adaptive Fusion](https://github.com/ShiQiu0419/BAAF-Net)

```python
k_n = 16  # KNN
num_layers = 5  # Number of layers
num_points = 65536  # Number of input points
num_classes = 11  # Number of valid classes
sub_grid_size = 0.06  # preprocess_parameter
# batch_size need to reset
batch_size = 1  # batch_size during training
val_batch_size = 1  # batch_size during validation and test
train_steps = 500  # Number of steps per epochs
val_steps = 100  # Number of validation steps per epoch

sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
d_out = [16, 64, 128, 256, 512]  # feature dimension

noise_init = 3.5  # noise initial parameter
# max_epoch need to reset
max_epoch = 100
learning_rate = 1e-2  # initial learning rate
lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

train_sum_dir = 'train_log'
saving = True
saving_path = None

augment_scale_anisotropic = True
augment_symmetries = [True, False, False]
augment_rotation = 'vertical'
augment_scale_min = 0.8
augment_scale_max = 1.2
augment_noise = 0.001
augment_occlusion = 'none'
augment_color = 0.8
```



### 1.5 SPG

[SPG: Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs](https://github.com/loicland/superpoint_graph)

```python
# Optimization arguments
wd = 0
lr = 1e-2
lr_decay = 0.7
lr_steps = '[350, 400, 450]'
momentum = 0.9
epochs = 500
batch_size = 1
optim = 'adam'
grad_clip = 1
loss_weights = 'none'
# Learning process arguments
cuda = 1
nworkers = 2
test_nth_epoch = 100
save_nth_epoch = 1
test_multisamp_n = 10
# Model
model_config = 'gru_10,f_8'
seed = 1
edge_attribs = 'delta_avg,delta_std,nlength/ld,surface/ld,volume/ld,size/ld,xyz/d'
# Point cloud processing
pc_attribs = 'xyzrgbelpsvXYZ'
pc_augm_scale = 0
pc_augm_rot = 1
pc_augm_mirror_prob = 0
pc_augm_jitter = 1
pc_xyznormalize = 1
# Filter generating network
fnet_widths = '[32,128,64]'
fnet_llbias = 0
fnet_orthoinit = 1
fnet_bnidx = 2
edge_mem_limit = 30000
# Superpoint graph
spg_attribs01 = 1
spg_augm_nneigh = 100
spg_augm_order = 3
spg_augm_hardcutoff = 512
spg_superedge_cutoff = -1
# Point net
ptn_minpts = 40
ptn_npts = 128
ptn_widths = '[[64,64,128,128,256], [256,64,32]]'
ptn_widths_stn = '[[64,64,128], [128,64]]'
ptn_nfeat_stn = 11
ptn_prelast_do = 0
ptn_mem_monger = 1
```



### 1.6 SQN

[SQN: Weakly-Supervised Semantic Segmentation of Large-Scale 3D Point Clouds](https://github.com/QingyongHu/SQN)

```python
k_n = 16  # KNN
num_layers = 5  # Number of layers
num_points = 65536  # Number of input points
num_classes = 11  # Number of valid classes
sub_grid_size = 0.06  # preprocess_parameter
# batch_size need to reset
batch_size = 1  # batch_size during training
val_batch_size = 1  # batch_size during validation and test
train_steps = 500  # Number of steps per epochs
val_steps = 100  # Number of validation steps per epoch

sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
d_out = [16, 64, 128, 256, 512]  # feature dimension

noise_init = 3.5  # noise initial parameter
# max_epoch need to reset
max_epoch = 100  # maximum epoch during training
learning_rate = 1e-2  # initial learning rate
lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

train_sum_dir = 'train_log'
saving = True
saving_path = None

augment_scale_anisotropic = True
augment_symmetries = [True, False, False]
augment_rotation = 'vertical'
augment_scale_min = 0.8
augment_scale_max = 1.2
augment_noise = 0.001
augment_occlusion = 'none'
augment_color = 0.8
```





