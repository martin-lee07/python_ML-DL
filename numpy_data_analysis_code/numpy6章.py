#深入学习numpy模块
#linalg模块
import numpy as np
a=np.matrix("0 1 2;1 0 3;4 -3 8")
print("A:",a)
#逆矩阵
print(np.linalg.inv(a))
#求解线性方程组
b=np.array([0,8,9])
print("b",b)
x=np.linalg.solve(a,b)
print("solution",x)
#验证
print("check:",np.dot(a,x))
#特征值eigenvalue,特征向量eigenvector
print("eigenvalue",np.linalg.eigvals(a))
print("eigenvector",np.linalg.eig(a))
#svg奇异值分解
u,sigma,v=np.linalg.svd(a,full_matrices=False)
print("u",u)
print("sigma",sigma)
print("v",v)
print("product",u*np.diag(sigma)*v)#真正的奇异值矩阵
#广义逆矩阵的求解摩尔彭罗斯广义逆矩阵可以使用numpy.linalg模块中的pinv函数进行计算
#计算广义逆矩阵需要用到奇异值分解，inv函数只接受方阵作为输入矩阵
pseudoinv=np.linalg.pinv(a)
print("pinv",pseudoinv)
#fft模块是一种快速计算离散傅里叶变换的模块，离散傅里叶级数在信号处理图像处理求解偏微分方程等方面具有重要意义
#有一个名为fft的模块能提供快速傅里叶变化的功能
#在这个模块中许多模块都是成对存在的，也就是说很多汗数据有对应的逆函数
#例如fft and ifft函数就是其中的一对
trnsformed=np.fft.fft(a)
from matplotlib.pyplot import plot,show
plot(trnsformed)
show()
