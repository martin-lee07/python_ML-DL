import numpy as np
from datetime import datetime
def datetonum(s):
    return datetime.strptime(s.decode('ascii'),"%d-%m-%Y").date().weekday()

dates,open,high,low,close=np.loadtxt('data.csv',delimiter=",",usecols=(1,3,4,5,6),converters={1:datetonum},unpack=True)
first_monday=np.ravel(np.where(dates==0))[0]
print("this is what we called the first monday",first_monday)
the_last_friday=np.ravel(np.where(dates==4))[-2]
print("this is what we called the last friday",the_last_friday)
week_indices=np.arange(first_monday,the_last_friday+1)
print("weeks initial ",week_indices)
week_indices2=np.split(week_indices,3)
print("splite into3 weeks",week_indices)

def summarize(a,o,h,l,c):
    monday_open=o[a[0]]
    week_high=np.max(np.take(h,a))
    week_low=np.min(np.take(l,a))
    friday_close=c[a[-1]]

    return("appl",monday_open,week_high,week_low,friday_close)
week_summarize=np.apply_along_axis(summarize,1,week_indices2,open,high,low,close)
print("week summaray",week_summarize)






print(np.arange(5))
a=np.array((np.arange(2,10),np.arange(2,10)))
print(a)

n=5
weights=np.ones((n,n))/n
print(weights)
print(np.full(np.array([1,2,3]),2))
print(np.ones([5,3]))
