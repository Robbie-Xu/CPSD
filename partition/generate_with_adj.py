#coding=utf-8
import pickle
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn import metrics
np.set_printoptions(threshold=np.inf)
import os
import json

# take the partition of nus-wide as an example.
num_classes = 81
cat2idx = json.load(open('category_nus.json', 'r'))
o = open("oc.txt",'w')
f = open("adj.txt",'w')
result = pickle.load(open("./adj.pkl", 'rb'))
_adj = np.array(result['adj'])
_nums = np.array(result['nums'])

for i in range(num_classes):
    o.write('\t'+str(list(cat2idx.keys())[i]))
o.write('\n')
for i,eachline in enumerate(_adj):
    o.write(str(list(cat2idx.keys())[i])+'\t')
    for each in eachline:
        o.write(str(each)+'\t')
    o.write('\n')
o.write(str(_nums))
o.close()

# DGP
_adj =1-np.power(_adj/_nums,1/2)
# CGP
# _adj =np.power(_adj/_nums,1/2)
_adj =(_adj+_adj.T)/2.0

for i in range(num_classes):
    f.write('\t'+str(list(cat2idx.keys())[i]))
f.write('\n')
for i,eachline in enumerate(_adj):
    f.write(str(list(cat2idx.keys())[i])+'\t')
    for each in eachline:
        f.write('{:.2f}'.format(each)+'\t')
    f.write('\n')
f.close()

# N-Cut implemention with sklearn
def turn_arg(X,k):
    scores = []
    s = dict()
    
    for index, gamma in enumerate((0.01, 0.1, 1, 10)):
        pred_y = SpectralClustering(n_clusters=k, gamma=gamma, affinity='precomputed',n_init=20).fit_predict(X)
        tmp = dict()
        tmp['gamma'] = gamma
        tmp['pred'] = pred_y
        tmp['score'] = metrics.calinski_harabasz_score(X, pred_y)
        s[metrics.calinski_harabasz_score(X, pred_y)] = tmp
        scores.append(metrics.calinski_harabasz_score(X, pred_y))

    pred = s.get(np.max(scores))['pred']
    return pred

def main():
    for j in range(5):
        k = 5
        X = _adj 

        pred_y = turn_arg(X,k)
        print("**************** cluster K="+str(k)+" ****************")
        print("best gamma-Calinski-Harabasz Score", metrics.calinski_harabasz_score(X, pred_y))
        print(pred_y)
    
        cat = list(cat2idx.keys())
        with open('./coocurrence'+str(k)+'_'+str(j)+'-.txt','w') as outf:
            for cluster in range(k):
                labels = []
                for i,id in enumerate(pred_y):
                    if id==cluster:
                        labels.append(cat[i])
                print("cluster "+str(cluster)+": "+ str(labels))
                outf.write(str(labels)+'\n')


if __name__ == '__main__':
    main()