import matplotlib.pyplot as plt

with open('log_abla.txt','r') as f:
    lines = f.readlines()

losses = []
bprs = []
cls = []
l1s = []
ndcgs = []
precs = []
recalls = []

for line in lines:
    words = line.strip().split(' ')
    if words[0][:5] == 'Epoch':
        if len(words[0]) > 5:
            losses.append(float(words[3]))
            bprs.append(float(words[5]))
            # l1s.append(float(words[7]))
            # cls.append(float(words[9]))
        else:
            ndcgs.append(float(words[3]))
            precs.append(float(words[5]))
            recalls.append(float(words[7]))

x_l = range(len(losses))
x_p = range(len(ndcgs))
plt.plot(x_l, losses, label='Overall Loss')
plt.plot(x_l, bprs, label='BPR Loss')
# plt.plot(x_l, l1s, label='L1 Loss')
# plt.plot(x_l, cls, label='InfoNCE Loss')
plt.legend()
plt.show()
plt.plot(x_p, ndcgs, label='NDCG@10')
plt.plot(x_p, precs, label='PREC@10')
plt.plot(x_p, recalls, label='RECALL@10')
plt.legend()
plt.show()
