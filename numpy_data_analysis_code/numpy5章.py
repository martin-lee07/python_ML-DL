#矩阵
import numpy as np
A=np.mat('0 1 2;1 0 3;4 -3 8')
print(A)
print(A.I)#inv函数取逆矩阵
print(np.mat(np.arange(9).reshape(3,3)))


#创建复合矩阵
print(np.bmat("A A;A A"))


a=np.arange(9)
print("a=",a)
#print(np.add(a))错误
print("reduce",np.add.reduce(a))#和reduce()类似，只是它返回的数组和输入数组的形状相同，保存所有的中间计算结果：
print("accumulate",np.add.accumulate(a))
print("reduceat",np.add.reduceat(a,[0,5,6,8]))#计算多组reduce()的结果，通过indices参数指定一系列的起始和终了位置
#print("at",np.negative.at(a, [0, 1]))


print(a.dtype)
print(a.itemsize)
#print(np.zeros((3,4))


c=np.random.random((2,3))#random记得是两个random
print("c:",c)
print("sum of c: ",c.sum())
print("axis was pointed,colume",c.sum(axis=0))# sum of each collume
print("axis was pointed,roll",c.sum(axis=1))#sum of each roll


def f(x,y,z):
    return 50*x+y+z
b = np.fromfunction(f,(2,4,3),dtype=int)#
#print(b)


print(b[...,2])
'''for row in b:
    print(row)'''


for i in b.flat:
    print(i)
print(b.resize())


for index,x in np.ndenumerate(b):#np.ndenumerate是将位置，值对显示
    print(index,x)
for index in np.ndindex(3,2):#仅为index
    print(index)

a=np.array([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
s=a[:,1:3]
print(s)

a = np.arange(12).reshape(3,4)
print(a)
i = np.array( [ [0,1],[0,1] ] )
j=np.array([[0,0],[0,0]])
print(a[i,j])
#divide()只保留整数
#true_divide()保留小数部分

#求对角元素二者相等
print(np.diag(a))
print(a.diagonal())

print(np.diagflat([1, 2, 3, 4]))#作为对角元素填充数组