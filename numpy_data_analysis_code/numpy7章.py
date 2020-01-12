import numpy as np 
import datetime
def datestr2(s):
    return datetime.datetime.strptime(s.decode("ascii"),"%d-%m-%Y").toordinal()
dates,closes=np.loadtxt('./data.csv',delimiter=',',usecols=(1,6),converters={1:datestr2},unpack=True)
print(dates)
indices=np.lexsort((dates,closes))#按收盘价排序
print(indices)

#np.random.seed(0)
complex_numbers=np.random.random(5)+1j*np.random.random(5)
print(complex_numbers)
print(np.sort_complex(complex_numbers))

a=np.array([2,3,4])
print(np.argmax(a))#返回下标

condition=(a%2)==0#取出
print(np.extract(condition,a))
print(np.nonzero(a))