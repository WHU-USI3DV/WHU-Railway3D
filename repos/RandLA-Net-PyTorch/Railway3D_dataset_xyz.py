from utils.helper_tool import DataProcessing as DP
from utils.helper_tool import ConfigRailway3D as cfg
from os.path import join
import numpy as np
import time, pickle, argparse, glob, os
from os.path import join
from utils.helper_ply import read_ply
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

def read_file_paths(file_path):
    with open(file_path, 'r') as file:
        paths = file.readlines()
    # 去除每个路径末尾的换行符
    paths = [path.rstrip() for path in paths]
    return paths

def add_prefix(paths, prefix):
    return [prefix + path for path in paths]

def split_filename(paths):
    return [path.split('/')[-1][:-4] for path in paths]

# read the subsampled data and divide the data into training and validation
class Railway3D(Dataset):
    def __init__(self):
        self.name = 'Railway3D'
        self.path   = 'repos/data/urban_railways/randla_processed_data'
        self.prefix = 'repos/data/urban_railways/'
        self.sub_pc_folder = join(self.path, 'input_{:.3f}'.format(cfg.sub_grid_size))

        self.label_to_names = {
                        0:  'rails',
                        1:  'track bed',
                        2:  'masts',
                        3:  'support devices',
                        4:  'overhead lines',
                        5:  'fences',
                        6:  'poles',
                        7:  'vegetation',
                        8:  'buildings',
                        9:  'ground', 
                        10: 'other'
                               }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.sort([])                                              # No ignored labels in this dataset

        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        cfg.class_weights = DP.get_class_weights('Railway3D')
        cfg.name = 'Railway3D'

        self.train_file_path_txt = 'repos/filelist/urban_railways/0-urban_train.txt'
        self.val_file_path_txt   = 'repos/filelist/urban_railways/0-urban_val.txt'
        self.test_file_path_txt  = 'repos/filelist/urban_railways/0-urban_test.txt'
        self.train_paths = read_file_paths(self.train_file_path_txt)
        self.val_paths = read_file_paths(self.val_file_path_txt)
        self.test_paths = read_file_paths(self.test_file_path_txt)
        self.train_paths = add_prefix(self.train_paths, self.prefix)
        self.val_paths = add_prefix(self.val_paths, self.prefix)
        self.test_paths = add_prefix(self.test_paths, self.prefix)
        self.all_filespath = self.train_paths + self.val_paths + self.test_paths
        self.train_file_name = split_filename(self.train_paths)
        self.val_file_name = split_filename(self.val_paths)
        self.test_file_name = split_filename(self.test_paths)
        self.cloud_names = self.train_file_name + self.val_file_name + self.test_file_name

        print('train_file_name = ', self.train_file_name)
        print('val_file_name = ', self.val_file_name)
        print('test_file_name = ', self.test_file_name)


        self.train_files = []
        self.val_files = []
        self.test_files = []
        # Split the dataset into training, validation and test sets based on filenames
        for full_file_path in self.all_filespath:
            pc_name = full_file_path.split('/')[-1][:-4]
            sub_file=join(self.sub_pc_folder, pc_name + '.ply') # Original version used for training and validation
            if pc_name in self.val_file_name:
                self.val_files.append(sub_file)
            elif pc_name in self.test_file_name:
                self.test_files.append(full_file_path)
            elif pc_name in self.train_file_name:
                self.train_files.append(sub_file)
            else:
                print('Not in file list: ', full_file_path)
        print("training files: ",self.train_files)
        print('####################################################################')
        print("val files: ",self.val_files)
        print('####################################################################')
        print("test files: ",self.test_files)

        self.all_files = self.all_filespath        # Get all ply files and return as a list

        self.size = len(self.all_files)

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.test_proj = []
        self.test_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': [], 'test': []}
        self.input_colors = {'training': [], 'validation': [], 'test': []}
        self.input_labels = {'training': [], 'validation': [], 'test': []}
        self.input_names = {'training': [], 'validation': [], 'test': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

        print('Size of training : ', len(self.input_colors['training']))                # Number of scenes in training set (112) (Area2-4)
        print('Size of validation : ', len(self.input_colors['validation']))            # Number of scenes in validation set (44) (Area1)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            print('Load_pc_' + str(i) + ': ' + cloud_name)
            if cloud_name in self.train_file_name:
                cloud_split = 'training'
            elif cloud_name in self.val_file_name:
                cloud_split = 'validation'
            elif cloud_name in self.test_file_name:
                cloud_split = 'test'
            else:
                print(f"Warning: File {cloud_name} not found in any split list.")
                continue

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))            # Load Sample data
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)                                                   # data['red'] is read as a 1D vector containing all color values
            # sub_colors = np.vstack((data['intensity'], data['intensity'], data['intensity'])).T
            sub_colors = np.ones((len(data['intensity']), 3))
            sub_labels = data['class']

            # Read pkl with search tree
            print('loading kd_tree_file: ', kd_tree_file)
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]              # List concatenation - input_trees dictionary stores two lists, each containing kdtree objects
            print('cloud_split = ', cloud_split)
            print('self.input_trees[cloud_split] = ', self.input_trees[cloud_split])
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices       # Used for projecting points back to original size during prediction
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if cloud_name in self.val_file_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]                 # Index of nearest points in subsampled cloud to original point cloud points
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

            if cloud_name in self.test_file_name:
                # tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.test_proj += [proj_idx]
                self.test_labels += [labels]


    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds
        return self.size


class Railway3DSampler(Dataset):

    def __init__(self, dataset, split='training'):
        self.dataset = dataset
        self.split = split
        self.possibility = {}
        self.min_possibility = {}

        # print('split = ', split)
        # print('self.input_trees = ', dataset.input_trees)
        # print('dataset.input_trees[split] = ', dataset.input_trees[split])
        # exit()

        if split == 'training':
            self.num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            self.num_per_epoch = cfg.val_steps * cfg.val_batch_size
        elif split == 'test':
            self.num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(dataset.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]              # Randomly generate possibility values for each point in every scene
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]         # Select the point with minimum possibility value in each scene

        # The probability calculation is used to randomly select center points in scenes. After selecting a center point,
        # K nearest points are found using KDTree (KNN). The possibility values of the center point and its neighbors 
        # are updated before feeding into the network to avoid repeated point selection. The possibility update adds
        # a value to the random initial value, where this added value is inversely proportional to the distance from
        # the center point (see line 146 in main_Railway3D). This update mechanism ensures previously selected points
        # have very low probability of being selected again, achieving an exhaustive-like sampling.

    def __getitem__(self, item):
        selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item, self.split)
        return selected_pc, selected_labels, selected_idx, cloud_ind

    def __len__(self):
        return self.num_per_epoch
        # return 2 * cfg.val_batch_size


    def spatially_regular_gen(self, item, split):
        # Select the scene containing the point with minimum possibility
        cloud_idx = int(np.argmin(self.min_possibility[split]))

        # Select point with minimum probability in the scene as query point, point_ind is the point index
        point_ind = np.argmin(self.possibility[split][cloud_idx])

        # Get xyz coordinates of all points in this scene from kdtree
        points = np.array(self.dataset.input_trees[split][cloud_idx].data, copy=False)

        # Select point with lowest probability from all points (using index calculated above), center_point shape is (1,3)
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)                    # Add noise

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < cfg.num_points:    # Take maximum of 40960 points (not all scenes have 40960 points, take all available points if less)
            # Query all points within the cloud
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

        # Randomly shuffle the indices
        queried_idx = DP.shuffle_idx(queried_idx)
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]            # Shuffle xyz information using list as index, where each number in list indexes matrix rows (first axis) and returns in order, used for matrix shuffling
        queried_pc_xyz = queried_pc_xyz - pick_point    # Subtract center point for centering
        queried_pc_colors = self.dataset.input_colors[split][cloud_idx][queried_idx]
        queried_pc_labels = self.dataset.input_labels[split][cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)    # Calculate distance from each point to center point
        delta = np.square(1 - dists / np.max(dists))    # Note multiplication/division before addition/subtraction. Cleverly calculate probability update magnitude (farther from center point = smaller probability increase = more likely to be selected as center in next iteration)
        self.possibility[split][cloud_idx][queried_idx] += delta    # Update probability to avoid selecting same center points in next iteration
        self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))  # Update minimum probability for this scene

        # up_sampled with replacement
        if len(points) < cfg.num_points:    # If less than 40960 points, use data augmentation to reach target count
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points) 


        queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()           # Convert back to tensor format
        queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
        queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
        queried_idx = torch.from_numpy(queried_idx).float() # keep float here?
        cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()

        points = torch.cat( (queried_pc_xyz, queried_pc_colors), 1)

        return points, queried_pc_labels, queried_idx, cloud_idx


    def tf_map(self, batch_xyz, batch_features, batch_label, batch_pc_idx, batch_cloud_idx):    # Perform downsampling and record KNN indices for subsequent network processing
        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):     # Implement downsampling for each layer (matrix order cannot be randomly shuffled from this point as KNN search relies on matrix indices to find neighbors)
            neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n)      # KNN search for 16 neighboring points around each point, record point indices, dimensions are (6, 40960, 16)
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]      # Random downsampling, dimensions are (6, 40960//4, 3)
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]      # Random downsampling for indices (6, 40960//4, 16)
            up_i = DP.knn_search(sub_points, batch_xyz, 1)                      # KNN search for nearest downsampled point to each original point, dimensions are (6, 40960, 1)
            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list

    # This function executes once for each data batch from dataloader
    def collate_fn(self,batch):

        selected_pc, selected_labels, selected_idx, cloud_ind = [],[],[],[]
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack(selected_pc)                     # Stack lists to form matrix, dimensions are (batch, nums, feature) = (6, 40960, 6)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        selected_xyz = selected_pc[:, :, 0:3]
        selected_features = selected_pc[:, :, 3:6]

        flat_inputs = self.tf_map(selected_xyz, selected_features, selected_labels, selected_idx, cloud_ind) # Returns a list containing 24 lists

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())     # Add five lists containing coordinates before each random sampling
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())    # Add five lists containing coordinates of 16 neighbors for input points before each random sampling (first list not downsampled)
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())      # Add five lists containing coordinates of 16 neighbors for input points after each random sampling
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())   # Add five lists containing nearest downsampled point for each original point after each random sampling

        # inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1,2).float()   # Transposed
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).float()  # Modified to match subsequent linear layer dimensions, no transpose needed
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()

        return inputs


if __name__ == '__main__':      # use to test
    dataset = Railway3D()
    dataset_train = Railway3DSampler(dataset, split='training')
    dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=dataset_train.collate_fn)
    # dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    for data in dataloader:

        features = data['features']
        labels = data['labels']
        idx = data['input_inds']
        cloud_idx = data['cloud_inds']
        print(features.shape)
        print(labels.shape)
        print(idx.shape)
        print(cloud_idx.shape)
        break

