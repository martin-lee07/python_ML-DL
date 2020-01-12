import numpy
from numpy import uint16
def pythonsum(n):
    a=list(range(n))
    b=list(range(n))
    c=[]
    for i in range(len(a)):
        a[i]=i**2
        b[i]=i**3
        c.append(a[i]+b[i])

    return c 

def numpysum(n):
    a=numpy.arange(n)**2
    b=numpy.arange(n)**3
    c=a+b
    return c

a=numpy.arange(5)
print(a.shape)
'''数组的shape属性返回了一个元组,元组中的元素多少既代表了numpy数组中每一个维度的大小'''

m=numpy.array([numpy.arange(2),numpy.arange(2)])
print(m)
print(m.shape)

b=numpy.array([[1,2],[3,4]])
print(b)
print(b.shape)
print(b[0,0])
'''too many indices ....这个错误是数组只有一组，但你取的有两组或两组以上的情况'''

'''复数不能转化成整数，但浮点数可以转化成复数'''
print(complex(45.5))

'''可以在函数中制定数据类型,但要先从numpy中导入'''
b=numpy.arange(7,dtype=uint16)
print(b)
print(b.dtype.itemsize)
'''单个数组元素在内存中战的字节数'''



'''切片'''
a=numpy.arange(9)
print(a[4:10])
'''选用2为步长选取元素'''
print(a[:7:2])
'''反转数组'''
print(a[::-1])

b=numpy.arange(24).reshape(2,3,4)
print(b.shape)
print(b)
print(b[1,0,0])
print(b[:,0,0])
print(b[::-1])
'''反转数组'''

'''方法：
ravel函数将数组展平
flatten函数展平并保存在内存
transpose转置矩阵
concatenate axis=1或hstack,把一个数组放到另一个数组后面
vshack垂直组合 vshack((a,b))
dshack深度组合 dshack((a,b))
hsplite横向切割切成列 hsplite(a,3) 把a切成三列
VSsplite纵向切割成行 vsplite(a,3) 把a切成三行
dsplite深度分割'''

'''属性：
ndmin数组维数
size数组中总数，
itemsize数组元素在内存中站的字节数
T与转置数组一个效果
real属性给出数组所有实部
imag给出数组所有虚部
tolist方法转换成列表
astype转换数组时指定数据类型  b.astype(int)'''
print(b.itemsize)
print(b.size)


