import torch
import pickle
import numpy as np
import torch.utils.data as data
from utils import scipy_sparse_mat_to_torch_sparse_tensor

def data_loader_hnd_handle(dataSetName,device):
    print(f'\nLoading dataset: {dataSetName}')
    path = 'data/' + f'{dataSetName}' + '/'

    # 加载训练集
    print('Processing train data...')
    f = open(path + 'trnMat.pkl', 'rb')
    train = pickle.load(f)  #coo_matrix  29601,24734  eg. (0,0) 0.0192

    rowSum = np.array(train.sum(1)).squeeze()
    colSum = np.array(train.sum(0)).squeeze()
    for i in range(len(train.data)):  # 按行列进行标准化
        train.data[i] = train.data[i] / pow(rowSum[train.row[i]] * colSum[train.col[i]], 0.5)

    #加载测试集
    print('Processing test data...')
    f = open(path + 'tstMat.pkl', 'rb')
    test = pickle.load(f)
    test_labels = [[] for i in range(test.shape[0])]
    for i in range(len(test.data)):
        row = test.row[i]
        col = test.col[i]
        test_labels[row].append(col)

    # 装入容器
    train = train.tocoo()
    dataContainer = DataContainer(train,test_labels,device)
    print('data processed.')

    return dataContainer


class DataContainer(data.Dataset):
    def __init__(self, train_mat, test_labels, device):
        self.rows = train_mat.row  #取行号
        self.cols = train_mat.col #取列号
        self.train_interaction = (train_mat != 0).astype(np.float32).todok() #dok是采用字典记录矩阵的
        self.row_negs = np.zeros(len(self.rows)).astype(np.int32) #行负样本
        self.col_negs = np.zeros(len(self.cols)).astype(np.int32)  # 列负样本
        self.adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train_mat).coalesce().cuda(torch.device(device))
        self.row_number = self.adj_norm.shape[0]
        self.col_number = self.adj_norm.shape[1]
        self.test_labels = test_labels

    def row_neg_sampling(self):  #负采样
        for i in range(len(self.rows)):
            r = self.rows[i]
            while True:
                neg = np.random.randint(self.train_interaction.shape[1])
                if (r, neg) not in self.train_interaction:
                    break
            self.row_negs[i] = neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.row_negs[idx]