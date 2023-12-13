import time
from dataset import *
from model import Model
import torch
from torch.optim import Adam
import utils
import numpy as np
import scipy.sparse as sp

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

A_id = pickle.load(open('user-item.pkl', 'rb'))
A = torch.sparse_coo_tensor(A_id, torch.ones((len(A_id[0]), )), size=(config.n, config.m)).to(device)

model = Model(config.n, config.m, config.d, vocab_size, config.h).to(device)
optimizer = Adam(model.parameters(), lr=config.lr)

test_id = pickle.load(open('test.pkl', 'rb'))
test_u = torch.tensor(list(test_id.keys())).long()
A_sp = sp.csr_matrix((np.ones(len(A_id[0])), (A_id[0], A_id[1])))
num_batches_test = int((len(test_id) - 1) / config.test_batch_size + 1)

if config.train:
    N = len(train)
    num_batches = int((N - 1) / config.batch_size + 1)
    best_ndcg = 0.0
    best_epoch = 0
    for epoch in range(config.n_epoch):
        model.train()
        for batch in range(0, num_batches):
            st = batch * config.batch_size
            end = min(st+config.batch_size, N)
            u, pos, neg, r, text = collate_batch(train, st, end)
            bpr, reg, cl, l1 = model(A, u.to(device), pos.to(device), neg, text.to(device), r.to(device))
            loss = bpr + l1 + config.cl * cl + config.reg * reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch{}/{} Batch{}/{} Loss {} BPR {} L1 {} CL {} Reg {}'
                  .format(epoch+1, config.n_epoch, batch+1, num_batches
                          , loss, bpr, l1, config.cl * cl, config.reg * reg))
        model.eval()
        ilist = {}
        for batch in range(0, num_batches_test):
            st = batch * config.test_batch_size
            end = min(st + config.test_batch_size, len(test_id))
            p = model.lightGCN.getScore(test_u[st:end]).cpu()
            for i in range(st, end):
                p[i - st, A_sp[test_u[i]].nonzero()[1]] = -np.inf
            items = torch.argsort(p, dim=-1, descending=True)[:, :config.k]
            for i in range(st, end):
                ilist[int(test_u[i])] = items[i - st, :].squeeze()
        ndcg = utils.NDCG_k(ilist, test_id, config.k)
        prec, recall = utils.precision_recall(ilist, test_id, config.k)
        print('Epoch {} NDCG@{} {} Prec@{} {} Recall@{} {}'.format(epoch+1, config.k, round(ndcg, 5)
                                                          , config.k, round(prec, 5), config.k, round(recall, 5)))
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_epoch = epoch+1
            torch.save(model.state_dict(), 'model_{}.pt'.format(epoch+1))
    print('Best Epoch ', best_epoch)

if not config.train:
    model.load_state_dict(torch.load('model_best.pt'))

ilist = {}
for batch in range(0, num_batches_test):
    st = batch * config.test_batch_size
    end = min(st + config.test_batch_size, len(test_id))
    p = model.lightGCN.getScore(test_u[st:end]).cpu()
    for i in range(st, end):
        p[i-st, A_sp[test_u[i]].nonzero()[1]] = -np.inf
    items = torch.argsort(p, dim=-1, descending=True)[:, :config.k]
    for i in range(st, end):
        ilist[int(test_u[i])] = items[i-st, :].squeeze()
ndcg = utils.NDCG_k(ilist, test_id, config.k)
prec, recall = utils.precision_recall(ilist, test_id, config.k)
print('NDCG@{} {} Prec@{} {} Recall@{} {}'.format(config.k, round(ndcg, 5)
                                                  , config.k, round(prec, 5), config.k, round(recall, 5)))
