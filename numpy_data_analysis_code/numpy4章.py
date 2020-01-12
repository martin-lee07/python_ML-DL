#便携函数
import numpy as np


#协方差函数cov()
c,v=np.loadtxt('./data.csv',delimiter=',',usecols=(3,4),unpack=True)
p=np.cov(c,v)
print(p)


#取对角线元素diagonal()
x=p.diagonal()
print(x)

#对角线元素之和trace()函数
d=p.trace()
print(d)

#相关系数矩阵corrcoef()
b=np.corrcoef(c,v)
print(b)

#多项式拟合polyfit()
v=np.polyfit(c,v,3)
print(v)