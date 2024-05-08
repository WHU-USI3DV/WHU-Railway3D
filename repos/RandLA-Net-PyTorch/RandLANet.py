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

        self.dilated_res_blocks = nn.ModuleList()       # LFA 编码器部分
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out                      # 乘以二是因为每次LFA的输出是2倍的dout(实际的输出feature的维度是2倍的dout)

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)       # 输入1024 输出1024的MLP（最中间的那层mlp）

        self.decoder_blocks = nn.ModuleList()       # 上采样 解码器部分
        for j in range(self.config.num_layers):
            # if j < 4:                                       
            #     d_in = d_out + 2 * self.config.d_out[-j-2]          # -2是因为最后一层的维度不需要拼接 乘二还是因为实际的输出维度是2倍的dout # din=1024+512 维度增加是因为进行了拼接
            #     d_out = 2 * self.config.d_out[-j-2]                 # 通过解码器里面的MLP调整回对应层的维度
            # else:
            #     d_in = 4 * self.config.d_out[-5]            # 第一个dout用了两次 4*16=64是因为64=32+32，由两个32进行拼接
            #     d_out = 2 * self.config.d_out[-5]           # 调整输出维度至32
            # self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))
            
            if j < config.num_layers - 1:                                       
                d_in = d_out + 2 * self.config.d_out[-j-2]          # -2是因为最后一层的维度不需要拼接 乘二还是因为实际的输出维度是2倍的dout # din=1024+512 维度增加是因为进行了拼接
                d_out = 2 * self.config.d_out[-j-2]                 # 通过解码器里面的MLP调整回对应层的维度
            else:
                d_in = 4 * self.config.d_out[-config.num_layers]            # 第一个dout用了两次 4*16=64是因为64=32+32，由两个32进行拼接
                d_out = 2 * self.config.d_out[-config.num_layers]           # 调整输出维度至32
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True))
            

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1,1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1,1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1,1), bn=False, activation=None)

    def forward(self, end_points):

        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)

        # 下面三行是后面改的
        features = self.fc0_acti(features)
        features = features.transpose(1,2)
        features = self.fc0_bath(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1 # 增加一个维度，是为了使用2d的[1,1]大小的卷积

        # ###########################Encoder############################
        f_encoder_list = []         # 用于保存每次LFA后的特征，方便后面进行拼接操作
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])    # 需要用到邻居的索引

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)      # 第一次把还没降采样时的也加上，feature维度为32，32在decoder用了两次
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])   # 中间那层MLP

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])                 # 先进行了插值
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))        # 和之前的特征进行拼接

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
    def random_sample(feature, pool_idx):       # 由于已经保存了索引值，所以随机采样只是读取索引值
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)    # batch*channel*npoints   # 减少一个维度
        num_neigh = pool_idx.shape[-1]      # knn邻居的数量
        d = feature.shape[1]                # 特征维数
        batch_size = pool_idx.shape[0]      # pool_idx的维度是[6, 10240, 16] 这个16是16个邻居的索引
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))  # 得到采样后点的特征
        # 这里先对pool_idx扩充一个中间的feature维度，扩充后为[batch, 1, npoints*nsamples]
        # 再在这个扩充的维度上将每个batch中的每一行的内容分别repeat feature.shape[1]-1 次（达到feature.shape[1]维） repeat后维度是 [batch, feature.shape[1], npoints*nsamples]
        # 然后根据这个处理之后的pool_idx进行feature张量的索引
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1  [0]是取值的意思 [1]是索引 max的意思是在每一维特征中取16近邻点中特征最大的特征
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
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1,feature.shape[1],1))  # 找到要上采样到的点的特征
        #（我觉得关键点在于数据矩阵的有序性，才可以将特征传播回原来上一次采样前的点）
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
        self.gt_classes = [0 for _ in range(cfg.num_classes)]               # 初始化一个长度为num_classes，元素全为0的列表
        self.positive_classes = [0 for _ in range(cfg.num_classes)]         # 同上  
        self.true_positive_classes = [0 for _ in range(cfg.num_classes)]
        self.cfg = cfg

    def add_data(self, end_points):
        logits = end_points['valid_logits']     # 忽略了label之后的logit        # 维度是（40960*batch_size）
        labels = end_points['valid_labels']     # 忽略了label之后的label
        pred = logits.max(dim=1)[1]             # [1] 是选择这个max对象的第二个位置，这个max对象长度为二，第一个位置存放取max之后的值，第二个位置存放max值的索引
        pred_valid = pred.detach().cpu().numpy()
        labels_valid = labels.detach().cpu().numpy()

        val_total_correct = 0       # 这个变量好像没什么用？
        val_total_seen = 0

        correct = np.sum(pred_valid == labels_valid)    # 计算分类正确的点数
        val_total_correct += correct    # 累加正确的点
        val_total_seen += len(labels_valid) # 累加一共的点

        # 计算混淆矩阵（混淆矩阵的列是预测类别，行是真实类别，描述的是正确分类和误分类的个数）
        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, self.cfg.num_classes, 1)) 
        self.gt_classes += np.sum(conf_matrix, axis=1)      # 按行加起来，表示某个类别一共有多少个真实的数据点（ground truth）
        self.positive_classes += np.sum(conf_matrix, axis=0)    # 按列加起来，表示某个类别被预测出多少个数据点
        self.true_positive_classes += np.diagonal(conf_matrix)  # 取出对角线上的元素

    def compute_iou(self):
        iou_list = []
        for n in range(0, self.cfg.num_classes, 1):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:       # 这里就是分母，保证分母不为零
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])  # 求第n个类的IoU
                iou_list.append(iou)
            else:
                iou_list.append(0.0)            # 三者同时为零才有可能分母为零，所以iou=0
        mean_iou = sum(iou_list) / float(self.cfg.num_classes)  # 除以类别数
        return mean_iou, iou_list



class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1,1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1,1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1                # 图中蓝色的那个MLP
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1      # 这个lfa包含两个局部空间编码和两个注意力池化
        f_pc = self.mlp2(f_pc)                                              # 后面的那个蓝色的MLP
        shortcut = self.shortcut(feature)                                   # 下面的那个MLP
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)              # 逐元素加起来


class Building_block(nn.Module):
    def __init__(self, d_out):  #  d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1,1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = pt_utils.Conv2d(d_out//2, d_out//2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10  # 这里10个feature是固定的
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples  # 交换张量的维度
        f_xyz = self.mlp1(f_xyz)            # 将空间特征进行编码,对应图中position encoding
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel 得到K个临近点的feature
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples 调整维度
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)      # 将特征信息和空间信息拼接在一起
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)        # 直接用上次编码好的空间信息再进行一次编码
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)  # batch*npoint*nsamples*channel
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples 调整维度
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3  这一步类似广播的操作，使得下一行可以直接相减 这一步的结果是中心点自己的xyz矩阵对应论文中的pi
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3   # 自己坐标减去近邻坐标计算相对坐标
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))  # batch*npoint*nsamples*1   # 离中心点的相对距离
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)  # batch*npoint*nsamples*10
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel(xyz或者feature)
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)      # 这个gather理解起来比较难，要多想想
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))     # 从原始点的xyz坐标（或feature）中，找到16个近邻点的坐标（或feature）（注意这个pc矩阵是有序的，其索引值和neighbor_idx有关系）
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel     # 这里就是40960个点中各个点的16近邻的坐标
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1,1), bn=True)     # 注意力池化里面还有一个mlp可以改变输出的形状

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)           # 将拼接后的矩阵经过一个全连接加softmax学习一个相同维度的注意力分数
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores                # 进行逐元素相乘
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)   # 求和
        f_agg = self.mlp(f_agg)                         # 最后一个mlp调整维度
        return f_agg


def compute_loss(end_points, cfg, device):

    logits = end_points['logits']       # 从网络中获取logit和label
    labels = end_points['labels']

    logits = logits.transpose(1, 2).reshape(-1, cfg.num_classes)        # 将logit和label的batch维度消去数据下放到点数目的维度
    labels = labels.reshape(-1)

    # Boolean mask of points that should be ignored
    # ignored_bool = labels == 0                              # 应该是这里0被mask了
    # for ign_label in cfg.ignored_label_inds:
    #     ignored_bool = ignored_bool | (labels == ign_label)

    # ignored_bool = labels == 0                              
    ignored_bool = torch.zeros(len(labels), dtype=torch.bool).to(device)
    for ign_label in cfg.ignored_label_inds:                                    # 这里没有问题，有问题的是后面
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
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)            # 这个操作没看懂
    loss = get_loss(valid_logits, valid_labels, cfg.class_weights, device)
    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels     # valid_logits是ignore label之后的logit
    end_points['loss'] = loss
    return loss, end_points


def get_loss(logits, labels, pre_cal_weights, device):
    # calculate the weighted cross entropy according to the inverse frequency
    class_weights = torch.from_numpy(pre_cal_weights).float().to(device)
    # one_hot_labels = F.one_hot(labels, self.config.num_classes)

    criterion = nn.CrossEntropyLoss(weight=class_weights.reshape([-1]), reduction='none')   # 这里改了一下维度，新版本pytorch需要一维的权重数据
    output_loss = criterion(logits, labels)
    output_loss = output_loss.mean()
    return output_loss
