import random
import numpy as np
import pandas as pd
import datetime

class TableNode(object):
    def __init__(self, index):
        self.val = index
        self.buckets = {}


def genPara(n, w):
    a = []
    for i in range(n):
        a.append(random.gauss(0, 1))
    b = random.uniform(0, w)

    return a, b


def gen_hash_function(n,k,w):
    result = []
    for i in range(k):
        result.append(genPara(n, w))

    return result


def gen_HashVals(hash_function, v, w):
    # hashVals include k values
    hashVals = []

    for hab in hash_function:
        hashVal = (np.inner(hab[0], v) + hab[1]) // w
        hashVals.append(hashVal)
    return hashVals


def H2(hashVals, fpRand, k, C):
    return int(sum([(hashVals[i] * fpRand[i]) for i in range(k)]) % C)


def train(dataSet, k, L, r, tableSize):
    #genrate hashtable
    hashTable = [TableNode(i) for i in range(tableSize)]
    #vector demonsion
    n = len(dataSet[0])
    m = len(dataSet)
    C = pow(2, 32) - 5
    hashFuncs = []
    fpRand = [random.randint(-10, 10) for i in range(k)]
    for times in range(L):
        hash_function = gen_hash_function(n, k, r)
        hashFuncs.append(hash_function)
        for dataIndex in range(m):
            # generate k hash values
            hashVals = gen_HashVals(hash_function, dataSet[dataIndex], r)
            # generate fingerprint
            fp = H2(hashVals, fpRand, k, C)
            # generate index
            index = fp % tableSize
            node = hashTable[index]
            if fp in node.buckets:
                bucket = node.buckets[fp]
                bucket.append(dataIndex+1000)
            else:
                node.buckets[fp] = [dataIndex+1000]
    return hashTable, hashFuncs, fpRand
def can_search(query, k, w, tableSize,temp):
    result = set()
    C = pow(2, 32) - 5
    hashTable = temp[0]
    hashFuncGroups = temp[1]
    fpRand = temp[2]
    for hashFuncGroup in hashFuncGroups:
        # H2
        queryFp = H2(gen_HashVals(hashFuncGroup, query, w), fpRand, k, C)
        # H1
        queryIndex = queryFp % tableSize
        # get the bucket in the dictionary
        if queryFp in hashTable[queryIndex].buckets:
            result.update(hashTable[queryIndex].buckets[queryFp])
    return result
data1 = pd.read_table('corel',header=None,sep=' ',engine='python')
data = np.array(data1)[:,1:]
temp = train(data[1000:,:], 5, 10, 0.05, 30)
candidates = []
results = []
starttime = datetime.datetime.now()
for index,i in enumerate(data):
    if(index==1000):
        break
    candidate = can_search(i,5,0.05,30,temp)
    candidates.append(candidate)
    nearest_neighbors = pd.DataFrame({'id': list(candidate)})
    candidates_data = data[np.array(list(candidate)).astype('int64'), :]
    nearest_neighbors['distance'] = [np.dot(i,vec2)/(np.linalg.norm(i)*np.linalg.norm(vec2)) for vec2 in candidates_data]
    result = nearest_neighbors.nlargest(10, 'distance')
    results.append(result.iloc[:, 0].to_list())
endtime = datetime.datetime.now()
print(endtime-starttime)
print(len(results))
results = np.array(results)-1000
print(results.tolist())
np.save('pretotal.npy',results)
