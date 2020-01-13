#1.导入numpy包命名为np
import numpy as np

#2.打印numpy版本和配置
print(np.__version__)
np.show_config()

#3.创建一个10*10的空数组
z = np.zeros(10)
print(z)

#4.查看数组内存大小
z = np.zeros((10, 10))
print(z)
print(z.size)
print(z.itemsize)
print("%d bytes" % (z.size * z.itemsize))

#5.在命令行下查看numpy add函数的文档
#%run `python -c "import numpy;numpy.info(numpy.add)"`

#6.创建一个长度为10的0数组，第5个数值为1
z = np.zeros(10)
z[4] = 1
print(z)

#7.创建一个值从1到49的数组
z = np.arange(10, 50)
print(z)

#反转数组
z = np.arange(50)
z = z[::-1]
print(z)

#创建一个从0~8的3*3矩阵
z = np.arange(9).reshape(3, 3)
print(z)

nz = np.nonzero((1, 2, 0, 0, 4, 0))
#从[1,2,0,0,4,0]中找到
#包装成ndarray
# nz=np.array(nz,dtype=float)
print(nz)

#生成一个3*3的对角矩阵
z=np.eye(3)
print(z)

#创建一个3*3*3的随机数组
z = np.random.random((3, 3, 3))
print(z)

#创建一个10*10的随机值数组，并找到最大值最小值
z = np.random.random(100).reshape(10, 10)
zmax, zmin = z.max(), z.min()
print(zmax,zmin)

#创建一个长度为30的随机值数组，并找到平均值
z=np.random.random(30)
m=z.mean()
print(m)

#创建一个中间为0，四边为1的二维数组
z = np.ones((10, 10))
z[1:-1,1:-1] = 0
print(z)

#如何给一个已经存在的数组添加边(填充0)
z=np.ones((5,5))
z=np.pad(z,pad_width=1,mode='constant',constant_values=0)
print(z)

#看看下列表达式的结果是什么
print(0*np.nan)#nan
print(np.nan == np.nan)  #False
print(np.inf>np.nan)#False
print(np.nan-np.nan)#nan
print(np.nan in set([np.nan]))#True
print(0.3 == 0.3 * 1)  #True

#创建一个5*5矩阵，对角线下方值为1，2，3，4
z=np.diag(1+np.arange(4),k=-1)
print(z)

#创建一个8*8矩阵，并用棋盘图案填充
z=np.zeros((8,8),dtype=int)
z[1::2,::2] = 1
z[::2, 1::2] = 1
print(z)
#广播机制
# for i in range(len(z)):
#     z[i]=z[i]+1
# print(z)

#给定一个（6，7，8）大小的三维矩阵，求100个元素的索引是什么
print(np.unravel_index(99,(6,7,8)))

#使用一个tile函数创建8*8的棋盘矩阵
#tile(A,resp)函数：A为原矩阵，resp为每个维度的重复次数
z=np.tile(np.array([[0,1],[1,0]]),(4,4))
print(z)

#对一个5*5矩阵进行标准化处理
#标准化：原值-平均值/方差
z=np.random.random((5,5))
z=((z-np.mean(z))/(np.std(z)))
print(z)

#新建立一个dtype类型用来描述一个颜色（RGBA）
color=np.dtype([("r",np.ubyte,1),("g",np.ubyte,1),("b",np.ubyte,1),("a",np.ubyte,1)])
print(color)

#5*3矩阵和3*2矩阵相乘
z=np.dot(np.ones((5,3)),np.ones((3,2)))
print(z)

z=np.ones((5,3))@np.ones((3,2))
print(z)

#给定一个一维数组，将3~8中间元素取反
z=np.arange(10)
z[(3<z)&(z<8)]*=-1
print(z)

#看看下列脚本会输出什么？
print(sum(range(5),-1))
from numpy  import *
print(sum(range(5),-1))

#给定一个整数数组z，看看下面哪个表达式是合法的？
z=[1,2,3,4]
# print(z**z)
# 2<<z>>2
# z<-z
# 1j*z
# z/1/1
# z<z>z

#下面表达式的结果是什么？
print(np.array(0)/np.array(0))#nan
print(np.array(0)//np.array(0))#0
print(np.array([np.nan]).astype(int).astype(float))#[-2.14748365e+09]

#np.random.uniform已...为分布采样
z=np.random.uniform(-10,+10,10)
print(z)
print(np.copysign(np.ceil(np.abs(z)),z))
