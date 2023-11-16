from torch import nn
import torch

class LightGCN(nn.Module):
    def __init__(self, n, m, d, nlayer):
        super(LightGCN, self).__init__()
        self.n = n
        self.m = m
        self.d = d
        self.nlayer = nlayer
        self.user_embeddings = nn.parameter.Parameter(torch.empty(n, d))
        self.item_embeddings = nn.parameter.Parameter(torch.empty(m, d))
        nn.init.xavier_normal_(self.user_embeddings)
        nn.init.xavier_normal_(self.item_embeddings)
        self.similarity = nn.CosineSimilarity(dim=1)
        self.linear = nn.Linear(2*d, d)

    def forward(self, A):
        u_embs = [self.user_embeddings]
        i_embs = [self.item_embeddings]
        for l in self.nlayer:
            u_embs.append(torch.sparse.mm(A, i_embs[l]))
            i_embs.append(torch.sparse.mm(A, u_embs[l]))
        self.user_embeddings = torch.mean(torch.stack(u_embs[1:], dim=0), dim=0)
        self.item_embeddings = torch.mean(torch.stack(i_embs[1:], dim=0), dim=0)
        return

    def getScore(self, u_idx):
        return nn.functional.sigmoid(self.user_embeddings[u_idx] @ self.item_embeddings.t())

    def bpr_loss(self, users, pos, neg):
        users_emb = self.user_embeddings[users]
        pos_emb = self.item_embeddings[pos]
        neg_emb = self.item_embeddings[neg]
        userEmb0 = self.user_embeddings[torch.unique(users)]
        posEmb0 = self.item_embeddings[torch.unique(pos)]
        negEmb0 = self.item_embeddings[torch.unique(neg)]
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss
