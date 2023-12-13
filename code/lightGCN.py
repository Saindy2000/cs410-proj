from torch import nn
import torch

class LightGCN(nn.Module):
    def __init__(self, n, m, d, nlayer=3):
        super(LightGCN, self).__init__()
        self.n = n
        self.m = m
        self.d = d
        self.nlayer = nlayer
        self.user_embeddings = nn.parameter.Parameter(torch.empty(n, d))
        self.item_embeddings = nn.parameter.Parameter(torch.empty(m, d))
        self.user_embeddings_forward = torch.empty(n, d)
        self.item_embeddings_forward = torch.empty(m, d)
        nn.init.xavier_uniform_(self.user_embeddings)
        nn.init.xavier_uniform_(self.item_embeddings)

    def forward(self, A, u_id, i_id):
        u_embs = [self.user_embeddings]
        i_embs = [self.item_embeddings]
        for l in range(self.nlayer):
            u_embs.append(torch.sparse.mm(A, i_embs[l]))
            i_embs.append(torch.sparse.mm(A.t(), u_embs[l]))
        self.user_embeddings_forward = torch.mean(torch.stack(u_embs, dim=1), dim=1)
        self.item_embeddings_forward = torch.mean(torch.stack(i_embs, dim=1), dim=1)
        return self.user_embeddings_forward[u_id], self.item_embeddings_forward[i_id]

    def getScore(self, u_idx):
        return self.user_embeddings[u_idx] @ self.item_embeddings.t()

    def bpr_loss(self, users, pos, neg):
        users_emb = self.user_embeddings_forward[users]
        pos_emb = self.item_embeddings_forward[pos]
        neg_emb = self.item_embeddings_forward[neg]
        userEmb0 = self.user_embeddings_forward[torch.unique(users)]
        posEmb0 = self.item_embeddings_forward[torch.unique(pos)]
        negEmb0 = self.item_embeddings_forward[torch.unique(neg)]
        reg_loss = (userEmb0.norm(2)/userEmb0.shape[0] 
                    + posEmb0.norm(2)/posEmb0.shape[0] 
                    + negEmb0.norm(2)/negEmb0.shape[0]) 
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = -torch.mean(torch.log(1e-12 + torch.sigmoid(pos_scores - neg_scores)))

        return loss, reg_loss
