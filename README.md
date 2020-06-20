# LSH
算法描述： 
1、基于P-table的LSH  
  第一步：生成hash系数  
  第二步：生成对应的a、b和r值  
  第三步：利用生成的a、b和r计算hashvalue
  第四步：根据已经计算的的hashvalue计算H2哈希值
  第五步：利用已有的值和tablesize计算已有的的buket
  第六步：利用已经训练的hash集计算最近邻的值
  第七步：计算查找值对应的hash值，利用这个计算对应的hashvalue和H2找到对应的bucket根据候选聚合规则计算所有的可信度的候选集
2、基于haming空间的LSH
  算法描述：  
 ```
class LSH:
    def __init__(self, data):
        self.data = data
        self.randvec = []
        self.binindex = []
        self.bucket = []
        self.hash_num=[]
 ```
num_hash:哈希表的数量，randvec,生成的随机系数，bindex：所有数据生成的hash下标，  
bucket生成的桶  
本算法包括下述几个函数：  
1、genrandom(num_hash,dim):  
		num_hash:生成hash_table的数量  
Dim：数据的维度  
Return randvec 生成用于划分线的系数  
2、train(lsh1, num_vector):  
Lsh1类，num_vector:划分直线数  
计算每一个点对于每一个bucket的位置，利用汉明空间，直线上和直线下  
3、simliar_bin 寻找相似点将寻找数据点转换为hanming距离在table中寻找已有的hamingbin，设定寻找step，每次变化一个位置的haming  
4、Query 利用similar寻找点后使用 heapd找出n大的值  
