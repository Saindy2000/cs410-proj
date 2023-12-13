import pickle

with open('user-item.pkl', 'rb') as f:
    train = pickle.load(f)

with open('test.pkl', 'rb') as f:
    test = pickle.load(f)

with open('train.txt', 'w') as f:
    for i in range(len(train[0])):
        f.write('{} {} 1\n'.format(train[0][i], train[1][i]))

with open('test.txt', 'w') as f:
    for u in test:
        for i in test[u]:
            f.write('{} {} 1\n'.format(u, i))
