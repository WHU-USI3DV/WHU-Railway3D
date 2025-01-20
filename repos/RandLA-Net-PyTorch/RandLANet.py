import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils
from helper_tool import DataProcessing as DP
import numpy as np
from sklearn.metrics import confusion_matrix


class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if(config.name == 'Railway3D'):
            self.class_weights = DP.get_class_weights('Railway3D')
            self.fc0 = nn.Linear(6, 8)
            self.fc0_acti = nn.LeakyReLU()
            self.fc0_bath = nn.BatchNorm1d(8, eps=1e-6, momentum=0.99)
            nn.init.constant_(self.fc0_bath.weight, 1.0)
            nn.init.constant_(self.fc0_bath.bias, 0)

        self.dilated_res_blocks = nn.ModuleList()       # LFA encoder section
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out                      # Multiply by 2, because each LFA outputs 2*dout features (actual output feature dimension is 2*dout)

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)       # Middle MLP layer with input/output dimension of 1024

        self.decoder_blocks = nn.ModuleList()       # Upsampling decoder section
        for j in range(self.config.num_layers):
            # if j < 4:
            #     d_in = d_out + 2 * self.config.d_out[-j-2]          # -2 because last layer dimension doesn't need concatenation. Multiply by 2 due to actual output dimension being 2*dout. din=1024+512 dimension increases due to concatenation
            #     d_out = 2 * self.config.d_out[-j-2]                 # Adjust dimensions back to corresponding layer through decoder MLP
            # else:
            #     d_in = 4 * self.config.d_out[-5]            # First dout used twice. 4*16=64 because 64=32+32, concatenated from two 32-dim features
            #     d_out = 2 * self.config.d_out[-5]           # Adjust output dimension to 32
            # self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))

            if j < config.num_layers - 1:
                d_in = d_out + 2 * self.config.d_out[-j-2]          # -2 because last layer dimension doesn't need concatenation. Multiply by 2 due to actual output dimension being 2*dout. din=1024+512 dimension increases due to concatenation
                d_out = 2 * self.config.d_out[-j-2]                 # Adjust dimensions back to corresponding layer through decoder MLP
            else:
                d_in = 4 * self.config.d_out[-config.num_layers]            # First dout used twice. 4*16=64, 64=32+32, concatenated from two 32-dim features
                d_out = 2 * self.config.d_out[-config.num_layers]           # Adjust output dimension to 32
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1,1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1,1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1,1), bn=False, activation=None)

    def forward(self, end_points):

        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)

        features = self.fc0_acti(features)
        features = features.transpose(1,2)
        features = self.fc0_bath(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1 # Add dimension to use 2D convolution with kernel size [1,1]

        # ###########################Encoder############################
        f_encoder_list = []         # Store features after each LFA for later concatenation operations
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])    # Requires neighbor indices

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)      # Add features before first downsampling, feature dimension is 32, used twice in decoder
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])   # Middle MLP layer

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])                 # Perform interpolation first
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))        # Concatenate with previous features

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)

        end_points['logits'] = f_out
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):       # Random sampling only reads index values since they are already saved
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)    # batch*channel*npoints   # Reduce dimension
        num_neigh = pool_idx.shape[-1]      # Number of KNN neighbors
        d = feature.shape[1]                # Feature dimension
        batch_size = pool_idx.shape[0]      # pool_idx dimension is [6, 10240, 16], where 16 represents indices of 16 neighbors
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))  # get the feature after pooling
        # First expand pool_idx with middle feature dimension to [batch, 1, npoints*nsamples]
        # Then repeat each row in batch feature.shape[1]-1 times along expanded dimension (to reach feature.shape[1] dim), 
        # The feature dimension after repeat is [batch, feature.shape[1], npoints*nsamples]
        # Finally index feature tensor using processed pool_idx
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]
        # batch*channel*npoints*1, [0] gets values, [1] gets indices.
        # max selects maximum feature value among 16 neighbors for each feature dimension.
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))  # Find the feature of the upsampled points
        # Key point is the ordered nature of data matrix, allowing feature propagation back to points before previous sampling
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features


def compute_acc(end_points):

    logits = end_points['valid_logits']
    labels = end_points['valid_labels']
    logits = logits.max(dim=1)[1]
    acc = (logits == labels).sum().float() / float(labels.shape[0])
    end_points['acc'] = acc
    return acc, end_points


class IoUCalculator:
    def __init__(self, cfg):
        # Initialize lists of length num_classes with zeros
        self.gt_classes = [0 for _ in range(cfg.num_classes)]
        self.positive_classes = [0 for _ in range(cfg.num_classes)]
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        # Get logits and labels after ignoring specified labels
        logits = end_points['valid_logits']     # Dimension is (40960*batch_size)
        labels = end_points['valid_labels']
        pred = logits.max(dim=1)[1]
        # Get index of maximum value, [1] selects the second element (indices) from max object
        # max object has length 2: first element contains max values, second contains indices
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0
        val_total_seen = 0

        # Calculate number of correctly classified points
        correct = np.sum(pred_valid == labels_valid)
        # Accumulate correct points
        val_total_correct += correct
        # Accumulate total points
        val_total_seen += len(labels_valid)

        # Calculate confusion matrix
        # Columns are predicted classes, rows are true classes, Describing the number of correct and incorrect classifications
        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.cfg.num_classes, 1))
        # Sum along rows: total number of ground truth points for each class
        self.gt_classes += np.sum(conf_matrix, axis=1)
        # Sum along columns: total number of predicted points for each class
        self.positive_classes += np.sum(conf_matrix, axis=0)
        # Diagonal elements: number of correctly predicted points for each class
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            # Check if denominator is non-zero
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                # Calculate IoU for class n
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
            else:
                # If all three values are zero, IoU is set to 0
                iou_list.append(0.0)
        # Calculate mean IoU by dividing sum by number of classes
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)
        return mean_iou, iou_list



class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        # Blue MLP in the figure(referring to the paper)
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        # This LFA contains two local spatial encodings and two attention poolings
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        # Blue MLP after LFA
        f_pc = self.mlp2(f_pc)
        # Shortcut connection
        shortcut = self.shortcut(feature)
        # Element-wise addition with leaky ReLU activation
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10  # Here 10 features are fixed dimensions
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples  # Permute tensor dimensions to [batch, 10, npoint, nsamples]
        f_xyz = self.mlp1(f_xyz)            # Encode spatial features, corresponding to position encoding in figure
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel, Get features of K nearest neighbors
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples, Adjust dimensions
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)      # Concatenate feature information and spatial information
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)        # Directly encode previously encoded spatial information again
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples, Adjust dimensions
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3   Broadcasting-like operation to enable direct subtraction in next line. Result is center point xyz matrix corresponding to pi in the paper
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3   # Calculate relative coordinates by subtracting neighbor coordinates from self coordinates
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1  # Calculate relative distance from center point
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel(xyz or feature)
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)      # This gather operation requires careful understanding
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        # Find coordinates (or features) of 16 nearest neighbors from original point xyz coordinates (or features).
        # Note: pc matrix is ordered, with indices corresponding to neighbor_idx
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)     # Attention pooling contains another MLP that can change output shape

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)           # Learn attention scores of same dimension through fully connected layer and softmax on concatenated matrix
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores                # Perform element-wise multiplication
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)   # Compute sum
        f_agg = self.mlp(f_agg)                         # Apply MLP to adjust dimensions
        return f_agg


def compute_loss(end_points, cfg, device):

    logits = end_points['logits']       # Get logits and labels from network output
    labels = end_points['labels']

    logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)        # Reshape logits and labels by merging batch dimension into point dimension
    labels = labels.reshape(-1)

    # Boolean mask of points that should be ignored
    # ignored_bool = labels == 0     # Zero labels are masked here
    # for ign_label in cfg.ignored_label_inds:
    #     ignored_bool = ignored_bool | (labels == ign_label)

    # ignored_bool = labels == 0
    ignored_bool = torch.zeros(len(labels), dtype=torch.bool).to(device)
    for ign_label in cfg.ignored_label_inds:    # This part works correctly, issue is in later steps
        ignored_bool = ignored_bool | (labels == ign_label)

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]

    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, cfg.num_classes).long().to(device)
    inserted_value = torch.zeros((1,)).long().to(device)
    for ign_label in cfg.ignored_label_inds:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    loss = get_loss(valid_logits, valid_labels, cfg.class_weights, device)
    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels     # valid_logits are logits after ignoring specified labels
    end_points['loss'] = loss
    return loss, end_points


def get_loss(logits, labels, pre_cal_weights, device):
    # calculate the weighted cross entropy according to the inverse frequency
    class_weights = torch.from_numpy(pre_cal_weights).float().to(device)
    # one_hot_labels = F.one_hot(labels, self.config.num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights.reshape([-1]), reduction='none')   # New version of pytorch requires one-dimensional weight data
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss
