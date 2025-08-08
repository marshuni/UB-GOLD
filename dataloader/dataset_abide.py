import numpy as np
import random
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.utils import degree
import torch.nn.functional as F

from torch_geometric.data import InMemoryDataset
class CustomDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(CustomDataset, self).__init__()
        self.data, self.slices = self.collate(data_list)

def construct_data(corr, label, threshold = 50):
    """
    create pyg data object from functional connectome matrix. We use correlation as node features
    Args:
    corr (n x n numpy matrix): functional connectome matrix
    Threshold (int (1- 100)): threshold for controling graph density. 
    the more higher the threshold, the more denser the graph. default: 50
    """
    
    A = torch.tensor(corr.copy())
    threshold = np.percentile(A[A > 0], 100 - threshold)
    A[A < threshold] = 0
    A[A >= threshold] = 1
    edge_index = A.nonzero().t().to(torch.long)
    data = Data(x = torch.tensor(corr, dtype=torch.float), edge_index=edge_index, y = label)

    # 计算每个节点的度数（边的起点节点是 edge_index[0]）,限制最大度数
    deg = degree(data.edge_index[0], data.num_nodes).long()
    deg = deg.clamp(max=10)
    
    # 生成 one-hot 编码，形状为 [num_nodes, max_degree + 1]
    one_hot_deg = F.one_hot(deg, num_classes=10 + 1).to(torch.float)

    # 对原始特征和度数的 one-hot 编码分别进行标准化
    data.x = (data.x - data.x.mean(dim=0, keepdim=True)) / (data.x.std(dim=0, keepdim=True) + 1e-6)
    one_hot_deg = (one_hot_deg - one_hot_deg.mean(dim=0, keepdim=True)) / (one_hot_deg.std(dim=0, keepdim=True) + 1e-6)

    data.x = torch.cat([data.x, one_hot_deg], dim=1)

    return data

def get_dataset(split):
    # data = np.load('data/ABIDE/abide.npy', allow_pickle=True).item()
    normal_graph_label = 0
    random_seed = 42
    labels = np.load('data/ABIDE/abide_label.npy')
    corr_matrices = np.load('data/ABIDE/abide_corr.npy')

    pyg_data_list = []
    for i in range(len(corr_matrices)):
        corr_matrix = corr_matrices[i]
        label = labels[i]
        graph = construct_data(corr_matrix, label=label, threshold=15)
        pyg_data_list.append(graph)

    dataset_raw = CustomDataset(pyg_data_list)
    
    class_labels = set([int(id) for id in dataset_raw.y])
    normal_indices = []
    anomaly_indices = []
    for label in class_labels:
        if label == normal_graph_label:
            normal_indices += (dataset_raw.y == label).nonzero().view(-1).tolist()
        else:
            anomaly_indices += (dataset_raw.y == label).nonzero().view(-1).tolist()
        
    normal_count = len(normal_indices)
    anomaly_count = len(anomaly_indices)

    random.seed(random_seed)
    random.shuffle(normal_indices)
    random.shuffle(anomaly_indices)

    if split == 'test':
        normal_indices = normal_indices[:int(normal_count * 0.2)]
        anomaly_indices = anomaly_indices[:int(anomaly_count * 0.2)]
    elif split == 'val':
        normal_indices = normal_indices[int(normal_count * 0.2):int(normal_count * 0.3)]
        anomaly_indices = anomaly_indices[int(anomaly_count * 0.2):int(anomaly_count * 0.3)]
    else:
        normal_indices = normal_indices[int(normal_count * 0.3):]
        anomaly_indices = anomaly_indices[int(anomaly_count * 0.3):]

    # if anomaly_ratio < 1.0:
    #     anomaly_count = min(int(len(normal_indices) * anomaly_ratio / (1 - anomaly_ratio)), len(anomaly_indices))
    #     anomaly_indices = anomaly_indices[:anomaly_count]
    
    # print(f"Anomaly_count:{anomaly_count}")
    # print(f"Nomal_count:{len(normal_indices)}")

    subset_indices = normal_indices + anomaly_indices
    random.seed(random_seed)
    random.shuffle(subset_indices)

    for i in range(len(subset_indices)):
        if subset_indices[i] in anomaly_indices:
            dataset_raw.y[subset_indices[i]] = 1
        else:
            dataset_raw.y[subset_indices[i]] = 0

    dataset = Subset(dataset_raw, subset_indices)

    # 分离训练/验证/测试集的标准化
    all_features = torch.cat([data.x for data in dataset], dim=0)
    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True) + 1e-3  # 增加平滑项

    for data in dataset:
        data.x = (data.x - mean) / std

    return dataset


