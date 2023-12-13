import math

def NDCG_k(y, y_true, k):
    avg_ndcg = 0.0
    for u in y_true.keys():
        idcg = 0.0
        dcg = 0.0
        for t in range(k):
            if y[u][t] in y_true[u]:
                dcg += 1.0/math.log(t+2, 2)
        for t in range(min(k, len(y_true[u]))):
            idcg += 1.0/math.log(t+2, 2)
        avg_ndcg += dcg/idcg
    avg_ndcg /= len(y_true)
    return avg_ndcg

def precision_recall(y, y_true, k):
    hits = {}
    for u in y_true.keys():
        items = y_true[u]
        predicted = y[u].squeeze().tolist()
        hits[u] = len(set(items).intersection(set(predicted)))
    prec = sum([hits[u] for u in hits])
    prec = prec / (len(hits) * k)
    recall_list = [hits[u]/len(y_true[u]) for u in hits]
    recall = sum(recall_list) / len(recall_list)
    return prec, recall