import pandas as pd
import numpy as np
import seaborn as sns
import cPickle
import matplotlib.pyplot as plt

sns.set_palette('colorblind')
sns.set(style="white")


def pickle_to_file(data, fname):
	fh = open(fname, 'w') 
	cPickle.dump(data, fh) 
	fh.close() 

def pickle_from_file(fname): 
    fh = open(fname, 'r') 
    data = cPickle.load(fh) 
    fh.close() 
    return data

data = pickle_from_file('database_tables.pkl')

fig, axs = plt.subplots(2,7, figsize=(14,4), sharex=True, sharey=True)
axs = axs.ravel()

for i in range (0,13):                   
	axs[i] = sns.regplot(x='x',y='y',data=data[i], ax=axs[i], fit_reg=False, color='#0072B2', scatter_kws={"s": 10})
	if i not in [0,7]:
		axs[i].set_ylabel('')
	if i not in [6,7,8,9,10,11,12,13]:
		axs[i].set_xlabel('')

plt.savefig('2DPlots.svg', bbox_inches='tight', dpi=300) 
plt.show()


fig, ax = plt.subplots()
ax = sns.violinplot(data = data, cut=0)
ax.set_xlabel('Table #')
ax.set_xticklabels(['1','2','3','4','5','6','7','8','9','10','11','12','13'])
ax.set_ylabel('x')
plt.savefig('1DPlots2.svg', bbox_inches='tight', dpi=300) 
plt.show()


for i in range (0,13):                   
	sns.kdeplot(data[i]['x'], label='Table'+str(i))

plt.xlabel('x')
plt.savefig('1DPlots.svg', bbox_inches='tight', dpi=300) 
plt.show()

