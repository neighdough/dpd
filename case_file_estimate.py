import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import os

os.chdir(('/home/nate/dropbox-caeser/Projects/'
    'ShelbyCounty/DPD/ScanCaseFiles/Admin/'))

df = pd.DataFrame()
df['pages'] = np.linspace(1,280, 20)
df['mins'] = np.linspace(1,30, 20)
df['mins2'] = np.linspace(1,45, 20)
f = lambda x: np.log(x)
f2 = lambda x: np.log(x**2)
df['l'] = f(df.mins)
df['l2'] = f2(df.mins2)
df['l_scale'] = minmax_scale(df.l, feature_range=(1,30))
df['l2_scale'] = minmax_scale(df.l2, feature_range=(1, 45))
fig, ax = plt.subplots()
fig.set_size_inches((6,4))
plt.plot(df.pages, df.mins, label='Scan All')
plt.plot(df.pages, df.l_scale, label='Unordered Selective Scan')
plt.plot(df.pages, df.l2_scale, label='Ordered Selective Scan')
ax.legend()
ax.set(xlabel='Number of Pages', ylabel='Minutes')
plt.savefig('remaining_estimate.jpg', dpi=300)
df.to_csv('estimate_time.csv', index=False)
