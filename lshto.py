from copy import copy
from itertools import combinations
import numpy as np
import pandas as pd
import heapq
import codecs,csv
import datetime
# create class LSH to storage data and dinclude
class LSH:
    def __init__(self, data):
        self.data = data
        self.randvec = []
        self.binindex = []
        self.table = []
        self.hash_num=[]

def genrandom(num_hash, dim):
    """
    #get random_vector to generate multiplied hash table
    :param num_hash: num of hash table
    :param dim: demension of data
    :return: randvec has num_hash ramdom vector
    """
    #return a sequence with Gaussian distribution
    randvec = np.random.randn(num_hash,dim)
    return randvec

def train(lsh1, num_hash):
    """
    #train a model which to help generate different key-vlaue
    :param num_vector:
    :return:
    """
    #train a model which to help generate different key-vlaue
    dim = lsh1.data.shape[1]
    randvec = genrandom(num_hash, dim)
    powers_of_two = 1 << np.arange(num_hash - 1, -1, -1)
    table = {}
    hamingbin = ((lsh1.data.dot(randvec.T) >= 0 )+0)
    bindex = hamingbin.dot(powers_of_two)
    for i, j in enumerate(bindex):
        if j not in table:
            table[j] = []
        table[j].append(i)
    lsh1.hash_num = num_hash
    lsh1.randvec = randvec
    lsh1.binindex = bindex
    lsh1.table = table
    return lsh1

def simliar_bins(lsh1, query_bin_bits, table, steplong):
    hash_num = lsh1.hash_num
    powers_of_two = 1 << np.arange(hash_num - 1, -1, -1)
    candidate_set = set()
    nearby_bin = query_bin_bits.dot(powers_of_two)
    if nearby_bin in table:
        candidate_set.update(table[nearby_bin])
    for different_bits in combinations(range(hash_num), steplong):
        alternate_bits = copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i] = not alternate_bits[i]
        nearby_bin = alternate_bits.dot(powers_of_two)
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])

    return candidate_set

def query(lsh1, query_vec, k, max_steplong):
    data = lsh1.data
    table = lsh1.table
    randvec = lsh1.randvec
    #compute query's hanmingbin
    hamingbin = ((query_vec.dot(randvec.T) >= 0)+0).flatten()
    # Search nearby bins and collect candidates
    candidate_set = set()
    for steplong in range(max_steplong + 1):
        candidate_set.update(simliar_bins(lsh1,hamingbin, table,steplong))
        if(len(candidate_set)>10):
            break
    # Sort candidates by their true distances from the query
    nearest_neighbors = pd.DataFrame({'id': list(candidate_set)})
    candidates = data[np.array(list(candidate_set)).astype('int64'), :]
    nearest_neighbors['distance'] = [np.dot(query_vec,vec2)/(np.linalg.norm(query_vec)*np.linalg.norm(vec2)) for vec2 in candidates]
    return nearest_neighbors.nlargest(k, 'distance')

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存成功")



#data=[[1,2,3,4],[4,5,6,7],[100,3,4,5],[-10,2,3,4],[1,2,3,4],[2,3,4,5],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
#print(np.array(data)[0,:])
data1 = pd.read_table('corel',header=None,sep=' ',engine='python')
data = np.array(data1)[:,1:]
lsh_init = LSH(np.array(data[1000:,:]))
num_of_randvec = 40
lsh_tranin = train(lsh_init,num_of_randvec)
results = []
starttime = datetime.datetime.now()
for i in data[0:1000,:]:
    resulti = query(lsh_tranin, i, 10, 5)
    #print(resulti.iloc[:,0].to_list())
    results.append(resulti.iloc[:,0].to_list())
endtime = datetime.datetime.now()
print(endtime-starttime)
result = np.array(results)
np.save('pretotal2.npy',result)
print(results)


