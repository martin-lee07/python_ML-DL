import numpy as np
i2=np.eye(2)
print(i2)
np.savetxt("eye.txt",i2)

#vwap=np.average(c,weights=v) 求加权平均值

c=63778,22323
print(np.mean(c))
'''求算术平均值用mean()函数,'''

my_new_array = np.zeros((5)) 
'''5个0的一维数组'''
print (my_new_array)

my_random_array = np.random.random((5))
'''创建随机数组'''
print(my_random_array)

'''创建二维0数组'''
a=np.zeros((2,3))
print(a)

"创建二维数组并显示"
a=np.array([[4,5],[5,6]])
print(a[0][1])
"提取一列"
b=a[:,1]
print(b)

matrix_product = a.dot(b) 
print ("Matrix Product = ", matrix_product)


y=np.arange(10).reshape(2,5)
x=np.arange(10)
a = np.array([[1, 3], [2, 4], [5, 6]])
print(a.shape[0])
print(a[0:0,0:2])

a = np.array([[11, 12, 13, 14, 15],
              [16, 17, 18, 19, 20],
              [21, 22, 23, 24, 25],
              [26, 27, 28 ,29, 30],
              [31, 32, 33, 34, 35]])
print(a[0,1:4])
print(a[1:4,0])
print(a[::2,::2])
print(type(a))
print(a.sum())
print(a.min())
print(np.arange(0,100,10))

#索引
indices=[1,2,3]
b=a[indices]
print(b)


'''import matplotlib.pyplot as plt

a = np.linspace(0, 2 * np.pi, 50)
b = np.sin(a)
plt.plot(a,b)
mask = b >= 0
plt.plot(a[mask], b[mask], 'bo')
mask = (b >= 0) & (a <= np.pi / 2)
plt.plot(a[mask], b[mask], 'go')
plt.show()'''

"where函数"
b=np.where(a<50)
print(b)

"full函数,2×2的全3数组"
np.full((2,2),3)

"np.ptp(x)计算x最大差值"
x=25,56
a=np.array([1,23,45,56])
list=[1,60,56]
print(np.ptp(x))#可以对数组使用
print(np.min(a))#可以对数组使用
print(a.min())#只对数组可用x.min(),列表不行
#print(list.min())错误！！！

'''遍历元组或列表，
不用for index in range(len(x)) print(x[index]),
用for val in enumerate（x）print(val)'''

c,v=np.loadtxt('data.csv',delimiter=',',usecols=(4,5),unpack=True)
print(np.ptp(c))
print(np.max(c))
print(np.median(c))#中位数
print(np.msort(c))
print(np.var(c))#方差=np.mean((c-c.mean())**2)
