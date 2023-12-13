import json
import pickle
import random

with open('Digital_Music_5.json', 'r') as f:
    lines = f.readlines()

dataset_nlp = []
dataset_rec = [[],[]]
test = {}
userId = {}
itemId = {}
ucnt = 0
icnt = 0
testcnt = 0
split = 0.3

for line in lines:
    data = json.loads(line)
    if "reviewerID" in data and "asin" in data and "overall" in data and "reviewText" in data:
        r = data["overall"]
        if r < 3.0: continue
        u = data["reviewerID"]
        i = data["asin"]
        t = data["reviewText"]
        if u not in userId:
            userId[u] = ucnt
            ucnt += 1
        if i not in itemId:
            itemId[i] = icnt
            icnt += 1
        if random.random() < split:
            if userId[u] not in test:
                test[userId[u]] = []
            test[userId[u]].append(itemId[i])
            testcnt += 1
        else:
            dataset_nlp.append([userId[u], itemId[i], (r-2)/3, t])
            dataset_rec[0].append(userId[u])
            dataset_rec[1].append(itemId[i])

# 16518 11794
# 115998
# 49621
print(len(userId), len(itemId))
print(len(dataset_nlp))
print(testcnt)

with open('user-item.pkl', 'wb') as f:
    pickle.dump(dataset_rec, f)

with open('texts.pkl', 'wb') as f:
    pickle.dump(dataset_nlp, f)

with open('test.pkl', 'wb') as f:
    pickle.dump(test, f)
