import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

tips=sns.load_dataset("tips")
sns.relplot(x="total_bill",y='tip',hue="size",style='smoker',data=tips)

'''current_palette=sns.color_palette()
sns.palplot(current_palette)   '''
df = pd.DataFrame(dict(time=np.arange(500),
                       value=np.random.randn(500).cumsum()))
g = sns.relplot(x="time", y="value", kind="line", data=df)
g.fig.autofmt_xdate()

fmri = sns.load_dataset("fmri")
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri);

plt.show()



