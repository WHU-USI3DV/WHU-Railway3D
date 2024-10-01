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
# LOG_DIR = os.path.join(LOG_DIR, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))      # 返回的是英国时间
LOG_DIR = os.path.join(LOG_DIR, FLAGS.dataset_name)
if not os.path.exists(LOG_DIR):               
    print(os.path.join(LOG_DIR))
    os.makedirs(os.path.join(LOG_DIR)) # 创建多级目录
log_file_name = 'test_log_Railway3d_'+ models_name + '.txt'
print('log_file_name = ', log_file_name)
LOG_FOUT = open(os.path.join(LOG_DIR, log_file_name), 'a')      # 追加写入模式

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

        self.test_probs = [np.zeros(shape=[l.shape[0], dataset.num_classes], dtype=np.float32) # 初始化一个全零矩阵,用于放入所有场景所有点的预测
                           for l in dataset.input_labels['test']]

    def test(self, dataset, num_vote=10):
        # Smoothing parameter for votes
        test_smooth = 0.95
        # Number of points per class in test set
        val_proportions = np.zeros(dataset.num_classes, dtype=np.float32)      # 长度为11的一个向量
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.test_labels]) # 统计每个类别有多少个点
                i += 1
        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_vote:
            stat_dict = {}
            net.eval() # set model to eval mode (for bn and dp)
            iou_calc = IoUCalculator(cfg)    
            for batch_idx, batch_data in enumerate(test_dataloader):
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

                stacked_probs = end_points['valid_logits']         # logit值，还未经过归一化
                stacked_labels = end_points['valid_labels']
                point_idx = end_points['input_inds'].cpu().numpy()
                cloud_idx = end_points['cloud_inds'].cpu().numpy()

                correct = torch.sum(torch.argmax(stacked_probs, axis=1) == stacked_labels)        # 计算准确预测的点数
                acc = (correct / float(np.prod(stacked_labels.shape))).cpu().numpy()             # 计算正确率
                # print('step' + str(step_id) + ' acc:' + str(acc))
                new_min = np.min(test_dataloader.dataset.min_possibility['test'])
                log_string('Epoch {:3d}, Step {:3d} end. Min possibility = {:.1f}, acc = {:.2f}'.format(epoch_id, step_id, new_min, acc))
                stacked_probs = torch.reshape(stacked_probs, [cfg.val_batch_size, cfg.num_points,
                                            cfg.num_classes])
                stacked_probs = F.softmax(stacked_probs, dim=2).cpu().numpy()
                stacked_labels = stacked_labels.cpu().numpy()

                for j in range(np.shape(stacked_probs)[0]):     # batchsize次（20次）循环，这个for没看懂，看懂了就知道每个场景下的点云正确率怎么来的了
                    probs = stacked_probs[j, :, :]      # 取这个batch下第j个的预测结果（分数）
                    p_idx = point_idx[j, :]             # 取这个batch下第j个的场景中，本次预测结果的点的序号
                    c_i = cloud_idx[j][0]               # 预测的结果来自第j个场景，c_i是该点云的编号
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs # 相当于是一个互补滤波？ 对预测分数进行更新（累加）这里应该就是vote的核心
                step_id += 1


            new_min = np.min(test_dataloader.dataset.min_possibility['test'])
            print('new_min = ', new_min)

            if True:
                last_min = new_min
                print('new_min = ', new_min)

                print('Saving clouds')
                # Show vote results (On subcloud so it is not the good values here)
                log_string('\nConfusion on sub clouds')
                confusion_list = []

                num_val = len(dataset.input_labels['test'])           # 验证区域有多少个场景

                for i_test in range(num_val):
                    probs = self.test_probs[i_test]                         # 取出第i_test个场景的vote后的结果
                    preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32) # 学习这种索引方式，索引中的数组的长度不一定要比被索引的小
                    labels = dataset.input_labels['test'][i_test]     # 拿到第i_test个场景对应label                      
                    # Confs
                    confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]   # 计算该场景下的混淆矩阵（13*13）并追加保存为列表
                # Regroup confusions
                C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)                 # 堆叠以后按列加起来 表示整个Area的混淆矩阵
                # Rescale with the right number of point per class      # 这里应该是根据正确点重新缩放混淆矩阵？
                C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)            # 混淆矩阵按行加起来就是每个类别分到的点数
                # Compute IoUs
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)
                log_string(s + '\n')
                # if int(np.ceil(new_min)) % 1 == 0:
                print('int(np.ceil(new_min)) = ', int(np.ceil(new_min)))
                # if int(np.ceil(new_min)) > 1:
                if int(np.ceil(new_min)) > 2:
                    # Project predictions
                    log_string('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    proj_probs_list = []
                    for i_val in range(num_val):
                        # Reproject probs back to the evaluations points
                        proj_idx = dataset.test_proj[i_val]                  # 取出第i_val个场景的原始点编号
                        probs = self.test_probs[i_val][proj_idx, :]         # 这里的编号很意思，跟之前生成这个test_proj有关。这一步其实就已经完成了采样后点云到原始点云之间的投影（结果预测）
                        proj_probs_list += [probs]                          # 将原始点云的预测结果保存在这个list中
                    # Show vote results
                    log_string('Confusion on full clouds')
                    confusion_list = []
                    for i_test in range(num_val):
                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)   # 根据结果（logit）求出最终的分类
                        # Confusion
                        labels = dataset.test_labels[i_test]     # 取出该场景的label
                        preds = np.array(preds)
                        preds = preds.reshape(np.shape(preds)[0], 1)
                        labels = np.array(labels)
                        labels = labels.reshape(np.shape(labels)[0], 1)
                        assert preds.shape == labels.shape, "preds and labels must have the same shape"

                        acc = np.sum(preds == labels) / len(labels) # 计算准确率
                        log_string(dataset.input_names['test'][i_test] + ' Acc:' + str(acc))
                        error = np.where(labels == preds, 0, 1)

                        confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]   # 计算混淆矩阵

                        name = dataset.input_names['test'][i_test] + '.ply'
                        test_file_name = dataset.prefix + name
                        test_data = read_ply(test_file_name)
                        xyz = np.vstack((test_data['x'], test_data['y'], test_data['z'])).T

                        cloud_colors = railway3d_label_to_color[preds.astype(np.uint8)]
                        cloud_colors = cloud_colors.reshape(np.size(preds, 0), 3)
                        save_dir = os.path.join(LOG_DIR, models_name)
                        if not os.path.exists(save_dir):
                            os.makedirs(os.path.join(save_dir))
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
