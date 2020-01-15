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

#np.random.uniform(A,B,C)以为A,B分布采样
z=np.random.uniform(-10,+10,10)
print(z)
print(np.copysign(np.ceil(np.abs(z)),z))

#如何找到两个数组中的相同值
z1=np.random.randint(0,10,10)
z2=np.random.randint(0,10,10)
print(np.intersect1d(z1, z2))

#如何忽略所有的numpy warnings(不建议这么做)？
#开启自杀模式
defaults=np.seterr(all="ignore")
z=np.ones(1)/0

#回到正常
_ = np.seterr(**defaults)

#同样的方法，用context manager
with np.errstate(divide='ignore'):
    z = np.ones(1) / 0

#回到正常
_ = np.seterr(**defaults)
    

#下列的表达式是否相等？
np.sqrt(-1) == np.emath.sqrt(-1)

#如何获取昨天，今天，与明天的日期
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')

#如何得到2016年7月份的所有日期
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)

#如何计算((A+B)*(-A/2))，替换原值
A = np.ones(3) * 1
B = np.ones(3) * 2
C = np.ones(3) * 3
np.add(A,B,out=A)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A, B, out=A)
print(A)

#如何从一个随机数组中取出其整数部分，用五种方法
z=np.random.uniform(0,10,10)
print(z-z%1)
print(np.floor(z))
print (np.ceil(z)-1)
print (z.astype(int))
print(np.trunc(z))

#创建一个5*5的矩阵，每行都是（0，4）
z = np.zeros((5,5))
z += np.arange(5)
print(z)

#用一个生成器函数生成10个整数，用它创建一个数组
def generate():
    for x in range(10):
        yield x 
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)
#生成器用例
info = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
a = (i for i in info)
for i in a:
    print(i)
#普通Fibonacci
def fib(max):
    n, a, b = 0, 0, 1
    while n<max:
        a, b = b, a + b
        n = n + 1
        print(a)
    return 'done'
a=fib(10)
print(fib(10))

#生成器写Fibonacci
#fibonacci数列
def fib0(max):
    n,a,b =0,0,1
    while n < max:
        yield b
        a,b =b,a+b
        n = n+1
    return 'done'
 
a = fib(10)
print(fib(10))

def fib1(max):
    n,a,b =0,0,1
    while n < max:
        yield b
        a,b =b,a+b
        n = n+1
    return 'done'
for i in fib(6):
    print(i)

#使用try except捕捉停止越界
# def fib2(max):
#     n,a,b =0,0,1
#     while n < max:
#         yield b
#         a,b =b,a+b
#         n = n+1
#     return 'done'
# g = fib(6)
# while True:
#     try:
#         x = next(g)
#         print('generator: ',x)
#     except StopIteration as e:
#         print("生成器返回值：",e.value)
#         break

#创建一个大小为10的向量，其值的大小从0~1
Z = np.linspace(0,1,11,endpoint=False)[1:]
print(Z)

#创建一个随机大小为10的数组并排序
z=np.random.random(10)
z.sort()
print(z)

#如何计算一个小数组的和，比sum()更快？
z=np.arange(20)
np.add.reduce(z)

#检查两个随机数组A,B是否相等
A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
#假设长度相同
equal = np.allclose(A,B)
print(equal)
#同时验证长度与元素是否相同
equal = np.array_equal(A,B)
print(equal)

#使数组只可读不可更改
z=np.zeros(10)
z.flags.writeable = False
# z[0]=1

#使一个(10*2)的表示笛卡尔坐标系的矩阵变成极坐标系
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)

#创建一个大小为10的随机数组将最大值替换为0
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)

#创建一个以X,Y为坐标系并以[0,1]X[0,1]覆盖的区域
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)

#给出两个数组X,Y创建柯西矩阵C (Cij =1/(xi - yj))
#np.linalg.det()计算行列式，但是这里要求数组的最后两个维度必须是方阵。
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))

#打印每个numpy标量类型的最小与最大
for dtype in [np.int8, np.int32, np.int64]:
   print(np.iinfo(dtype).min)
   print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps)

#如何打印数组中所有元素
np.set_printoptions(threshold=10000)#输出数组的时候完全输出
Z = np.zeros((16,16))
print(Z)

#如何找到在一个向量中一个标量的最近值
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
