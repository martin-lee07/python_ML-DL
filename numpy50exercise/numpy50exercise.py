#1.导入numpy包命名为np
import numpy as np

#打印numpy版本和配置
print(np.__version__)
np.show_config()

#创建一个10*10的空数组
z = np.zeros(10)
print(z)

#查看数组内存大小
z = np.zeros((10, 10))
print(z)
print(z.size)
print(z.itemsize)
print("%d bytes" % (z.size * z.itemsize))

#在命令行下查看numpy add函数的文档
#%run `python -c "import numpy;numpy.info(numpy.add)"`

#创建一个长度为10的0数组，第5个数值为1
z = np.zeros(10)
z[4] = 1
print(z)

#创建一个值从1到49的数组
z = np.arange(10, 50)
print(z)

#反转数组
z = np.arange(50)
z = z[::-1]
print(z)

#创建一个从0~8的3*3矩阵
z = np.arange(9).reshape(3, 3)
print(z)

nz = np.nonzero([1, 2, 0, 0, 4, 0])
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

#