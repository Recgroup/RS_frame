import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F

class LightGCL(nn.Module):
    # def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device):
    def __init__(self, args, row_nmber, col_number, device):
        super(LightGCL, self).__init__()
        # rowNumber 代表邻接矩阵的行数
        # colNumber 代表邻接矩阵的列数
        #  u_mul_s, v_mul_s,  分解出来的两个矩阵各自与S的乘积
        # ut, vt,  分解出来的两个矩阵
        # train_csr  原始的训练数据
        #adj_norm 归一化邻接矩阵
        # l gnn层数  temp
        dim = args.dim
        self.layer = args.gnn_layer
        self.temp = args.temp
        self.lambda_1 = args.lambda1
        self.lambda_2 = args.lambda2
        self.dropout = args.dropout
        self.act = nn.LeakyReLU(0.5)
        self.svd_q = args.svd_q
        self.device = device

        #生成可训练嵌入
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(row_nmber, dim)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(col_number, dim)))

        #全嵌入list
        self.E_u_list = [None] * (self.layer+1)
        self.E_i_list = [None] * (self.layer+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0

        #重构的图的嵌入
        self.Z_u_list = [None] * (self.layer+1)
        self.Z_i_list = [None] * (self.layer+1)

        #原图的嵌入·
        self.G_u_list = [None] * (self.layer+1)
        self.G_i_list = [None] * (self.layer+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0


        self.E_u = None
        self.E_i = None

        self.u_mul_s = None
        self.v_mul_s = None
        self.ut = None
        self.vt = None

    def svd_reconstruction(self,adj, svd_q):
        svd_u, s, svd_v = torch.svd_lowrank(adj, q=svd_q)
        u_mul_s = svd_u @ (torch.diag(s))
        v_mul_s = svd_v @ (torch.diag(s))
        return u_mul_s, v_mul_s, svd_u.T, svd_v.T

    def get_embedding(self):
        return self.E_u,self.E_i

    def forward(self, adj, row_ids, col_ids, pos, neg):

        # svd 重构
        if self.v_mul_s==None and self.u_mul_s==None and self.vt==None and self.ut==None:
            self.u_mul_s, self.v_mul_s, self.ut, self.vt = self.svd_reconstruction(adj,self.svd_q)

        # train phase
        if True:

            for layer in range(1,self.layer+1):
                # GNN propagation
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(adj,self.dropout), self.E_i_list[layer-1]))
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(adj,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

                # svd_adj propagation
                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            neg_score = torch.log(torch.exp(G_u_norm[row_ids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[col_ids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[row_ids] * E_u_norm[row_ids]).sum(1) / self.temp, -5.0, 5.0)).mean() + (torch.clamp((G_i_norm[col_ids] * E_i_norm[col_ids]).sum(1) / self.temp, -5.0, 5.0)).mean()
            loss_cl = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[row_ids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_bpr = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()

            # total loss
            loss = loss_bpr + self.lambda_1 * loss_cl + self.lambda_2 * loss_reg
            other_information = torch.tensor([loss_bpr, self.lambda_1 * loss_cl])

            return loss, other_information  # other_information size必须是（x,）
