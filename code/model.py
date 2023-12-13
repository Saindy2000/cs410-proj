import torch
from torch import nn
from transformer import Transformer
from lightGCN import LightGCN

class Model(nn.Module):
    def __init__(self, n, m, d, ntoken, d_hidden):
        super(Model, self).__init__()
        self.lightGCN = LightGCN(n, m, d)
        self.Transformer = Transformer(ntoken, d, d_hidden)

    def forward(self, A, u_id, pos_id, neg_id, T, r_true):
        u_emb, i_emb, r = self.Transformer(T)
        u_emb1, i_emb1 = self.lightGCN(A, u_id, pos_id)
        bpr_loss, reg_loss = self.lightGCN.bpr_loss(u_id, pos_id, neg_id)
        reg_loss1 = 1/2*(u_emb.norm(2)+i_emb.norm(2))/len(u_id)
        return bpr_loss, reg_loss+reg_loss1, self.cl_loss(u_emb, i_emb, u_emb1, i_emb1)\
            , nn.functional.l1_loss(r, r_true, reduction='mean')

    def cl_loss(self, u_emb, i_emb, u_emb1, i_emb1, tau=0.2):
        u_emb = nn.functional.normalize(u_emb, dim=1)
        i_emb = nn.functional.normalize(i_emb, dim=1)
        u_emb1 = nn.functional.normalize(u_emb1, dim=1)
        i_emb1 = nn.functional.normalize(i_emb1, dim=1)
        s_u = u_emb.multiply(u_emb1) / tau
        s_i = i_emb.multiply(i_emb1) / tau
        t_u = u_emb.matmul(u_emb1.t()) / tau
        t_i = i_emb.matmul(i_emb1.t()) / tau
        s_u = torch.exp(s_u) / torch.exp(t_u).sum(dim=1).unsqueeze(dim=1)
        s_i = torch.exp(s_i) / torch.exp(t_i).sum(dim=1).unsqueeze(dim=1)
        l = -torch.log(s_u).mean() - torch.log(s_i).mean()
        return l
