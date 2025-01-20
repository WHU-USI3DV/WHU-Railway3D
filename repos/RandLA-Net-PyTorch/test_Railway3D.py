from utils.helper_tool import ConfigRailway3D as cfg
from RandLANet import Network, compute_loss, compute_acc, IoUCalculator
from Railway3D_dataset import Railway3D, Railway3DSampler # when training with xyz + intensity
# from Railway3D_dataset_xyz import Railway3D, Railway3DSampler # when training with only xyz
import numpy as np
import os, argparse
# from os.path import dirname, abspath, join

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from utils.helper_tool import DataProcessing as DP
from utils.helper_ply import read_ply, write_ply
from datetime import datetime
import time

railway3d_label_to_color = np.asarray([
    [154,107,56]   ,
    [160,112,160]  ,
    [251,143,49]   ,
    [62,170,163]   ,
    [229,202,63]   ,
    [101,101,206]  ,
    [180,47,174]   ,
    [166,255,64]   ,
    [253,51,103]   ,
    [154,194,231]  ,
    [218,219,220]
])

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='train_output/xyz_11class_2024-04-20_17-40-28/checkpoint.tar', help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='test_output', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--gpu', type=int, default=0, help='which gpu do you want to use [default: 2], -1 for cpu')
parser.add_argument('--dataset_name', default='Railway3D', help='dataset name')
FLAGS = parser.parse_args()
#################################################   log   #################################################

models_name = FLAGS.checkpoint_path
models_name = models_name.split('/')[-2]
print('models_name = ', models_name)

LOG_DIR = FLAGS.log_dir
# LOG_DIR = os.path.join(LOG_DIR, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))      # Returns UTC time
LOG_DIR = os.path.join(LOG_DIR, FLAGS.dataset_name)
if not os.path.exists(LOG_DIR):
    print(os.path.join(LOG_DIR))
    # Create multi-level directories
    os.makedirs(os.path.join(LOG_DIR))
log_file_name = 'test_log_Railway3d_'+ models_name + '.txt'
print('log_file_name = ', log_file_name)
# Append write mode
LOG_FOUT = open(os.path.join(LOG_DIR, log_file_name), 'a')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

dataset = Railway3D()
test_dataset = Railway3DSampler(dataset, 'test')
test_dataloader = DataLoader(test_dataset, batch_size=cfg.val_batch_size, shuffle=True, collate_fn=test_dataset.collate_fn)

if FLAGS.gpu >= 0:
    if torch.cuda.is_available():
        FLAGS.gpu = torch.device(f'cuda:{FLAGS.gpu:d}')
    else:
        warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
        FLAGS.gpu = torch.device('cpu')
else:
    FLAGS.gpu = torch.device('cpu')
device = FLAGS.gpu

net = Network(cfg)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)

checkpoint_path = FLAGS.checkpoint_path
print(os.path.isfile(checkpoint_path))
if checkpoint_path is not None and os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model restored from %s" % checkpoint_path)
else:
   raise ValueError('CheckPointPathError')


#################################################   test function   ###########################################
class ModelTester:
    def __init__(self, dataset):
        # Initialize prediction probabilities for each test scene
        self.test_probs = [np.zeros(shape=[l.shape[0], dataset.num_classes], dtype=np.float32)
                           for l in dataset.input_labels['test']]

    def test(self, dataset, num_vote=10):
        # Smoothing parameter for votes
        test_smooth = 0.95
        # Number of points per class in test set
        val_proportions = np.zeros(dataset.num_classes, dtype=np.float32)      # A vector of length 11 (number of classes)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.test_labels]) # Count number of points for each class
                i += 1
        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_vote:
            stat_dict = {}
            net.eval() # set model to eval mode (for bn and dp)
            iou_calc = IoUCalculator(cfg)
            for batch_idx, batch_data in enumerate(test_dataloader):
                # Move data to device
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(len(batch_data[key])):
                            batch_data[key][i] = batch_data[key][i].to(device)
                    else:
                        batch_data[key] = batch_data[key].to(device)

                # Forward pass
                with torch.no_grad():
                    end_points = net(batch_data)
                loss, end_points = compute_loss(end_points, cfg, device)

                stacked_probs = end_points['valid_logits']
                stacked_labels = end_points['valid_labels']
                point_idx = end_points['input_inds'].cpu().numpy()
                cloud_idx = end_points['cloud_inds'].cpu().numpy()

                correct = torch.sum(torch.argmax(stacked_probs, axis=1) == stacked_labels)        # Calculate number of correctly predicted points
                acc = (correct / float(np.prod(stacked_labels.shape))).cpu().numpy()             # Calculate accuracy
                # print('step' + str(step_id) + ' acc:' + str(acc))
                new_min = np.min(test_dataloader.dataset.min_possibility['test'])
                log_string('Epoch {:3d}, Step {:3d} end. Min possibility = {:.1f}, acc = {:.2f}'.format(epoch_id, step_id, new_min, acc))
                stacked_probs = torch.reshape(stacked_probs, [cfg.val_batch_size, cfg.num_points,
                                            cfg.num_classes])
                stacked_probs = F.softmax(stacked_probs, dim=2).cpu().numpy()
                stacked_labels = stacked_labels.cpu().numpy()

                # Loop batch_size times (20 times) - this loop updates prediction probabilities for each scene
                for j in range(np.shape(stacked_probs)[0]):
                    # Get prediction scores for the j-th sample in this batch
                    probs = stacked_probs[j, :, :]
                    # Get point indices for predictions in the j-th scene of this batch
                    p_idx = point_idx[j, :]
                    # Predictions come from j-th scene, c_i is the point cloud ID
                    c_i = cloud_idx[j][0]
                    # Acts like a complementary filter - updates prediction scores (accumulation). This is the core of voting mechanism
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                step_id += 1

            new_min = np.min(test_dataloader.dataset.min_possibility['test'])

            if True:
                last_min = new_min

                print('Saving clouds')
                # Show vote results (On subcloud so it is not the good values here)
                log_string('\nConfusion on sub clouds')
                confusion_list = []

                # Number of scenes in validation area
                num_val = len(dataset.input_labels['test'])

                for i_test in range(num_val):
                    probs = self.test_probs[i_test]                         # Get voting results for i_test-th scene
                    preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32) # Note this indexing method - length of indexing array doesn't need to be smaller than indexed array
                    labels = dataset.input_labels['test'][i_test]     # Get labels for i_test-th scene
                    # Confs
                    confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]   # Calculate confusion matrix (13*13) for this scene and append to list
                # Regroup confusions
                # Stack and sum along columns to get confusion matrix for entire area
                C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)
                # Rescale confusion matrix based on correct point proportions
                C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
                # Compute IoUs
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                log_string(s + '\n')
                # if int(np.ceil(new_min)) > 1:
                if int(np.ceil(new_min)) > 2:
                    # Project predictions
                    log_string('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    proj_probs_list = []
                    for i_val in range(num_val):
                        # Reproject probs back to the evaluations points
                        proj_idx = dataset.test_proj[i_val]                 # Get original point indices for i_val-th scene
                        probs = self.test_probs[i_val][proj_idx, :]         # These indices are related to previously generated test_proj. This step completes projection of predictions from subsampled to original point cloud
                        proj_probs_list += [probs]                          # Store predictions for original point cloud in this list
                    # Show vote results
                    log_string('Confusion on full clouds')
                    confusion_list = []
                    for i_test in range(num_val):
                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)   # Determine final classification based on logits

                        # Confusion
                        labels = dataset.test_labels[i_test]     # Get labels for this scene
                        preds = np.array(preds).reshape(-1, 1)
                        labels = np.array(labels).reshape(-1, 1)
                        assert preds.shape == labels.shape, "preds and labels must have the same shape"

                        acc = np.sum(preds == labels) / len(labels) # Calculate accuracy
                        log_string(dataset.input_names['test'][i_test] + ' Acc:' + str(acc))
                        error = np.where(labels == preds, 0, 1)

                        confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]   # Calculate confusion matrix for this scene

                        name = dataset.input_names['test'][i_test] + '.ply'
                        test_file_name = dataset.prefix + name
                        test_data = read_ply(test_file_name)
                        xyz = np.vstack((test_data['x'], test_data['y'], test_data['z'])).T

                        cloud_colors = railway3d_label_to_color[preds.astype(np.uint8)]
                        cloud_colors = cloud_colors.reshape(np.size(preds, 0), 3)
                        save_dir = os.path.join(LOG_DIR, models_name)
                        if not os.path.exists(save_dir):
                            os.makedirs(os.path.join(save_dir))
                        submission_dir = os.path.join(save_dir, 'submission')
                        if not os.path.exists(submission_dir):
                            os.makedirs(submission_dir)
                        # Save ascii preds
                        ascii_name = os.path.join(submission_dir, dataset.input_names['test'][i_test] + '.npy')
                        np.savetxt(ascii_name, preds, fmt='%d')

                        pred_file = os.path.join(save_dir, name)
                        write_ply(pred_file, [xyz.astype(np.double), cloud_colors.astype(np.uint8),
                                              labels.astype(np.uint8), preds.astype(np.uint8), error.astype(np.uint8)],
                                              ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'pred', 'error'])

                    # Regroup confusions
                    C = np.sum(np.stack(confusion_list), axis=0)

                    IoUs = DP.IoU_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    log_string('-' * len(s))
                    log_string(s)
                    log_string('-' * len(s) + '\n')
                    print('finished \n')
                    return

            epoch_id += 1
            step_id = 0
            continue

        return


if __name__ == '__main__':
    test_model = ModelTester(dataset)
    test_model.test(dataset)
