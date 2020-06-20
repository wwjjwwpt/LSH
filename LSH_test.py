import numpy as np
import pandas as pd
a= np.load('pretotal2.npy').tolist()
c= np.load('pretotal.npy').tolist()
#print(a)
b = np.load('demo.npy')
def test(a,b):
    TP = []
    for index,i in enumerate(a) :
        num = 0
        for j in i:
            if(j in b[index]):
                num+=1
        TP.append(num)
    #print(TP)
    TP = np.array(TP)
    FP = 10 - TP
    FN = 10 - TP
    recall = (TP/10).mean()
    print("recall:",recall)
    data1 = pd.read_table('corel',header=None,sep=' ',engine='python')
    N = len(data1)-1010
    TN = N - FN
    ACCden = TP+TN
    ACCen = TP+TN+FP+FN
    Acc = ACCden/ACCen
    print("ACC:",Acc.mean())

test(a,b)
print('Phash')
test(c,b)
