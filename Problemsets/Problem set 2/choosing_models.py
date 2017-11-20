import cPickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as op

sns.set()
a = np.zeros(10)

def pickle_from_file(fname): 
    fh = open(fname, 'r') 
    data = cPickle.load(fh) 
    fh.close() 
    return data

def calc_polynomial(theta, x):
    """
    Calculate a polynomial function.
    """
    return np.polyval(theta[::-1], x)

def neglnL(theta, x, y, yerr, poly): 
	model = calc_polynomial(theta, x)
	inv_sigma2 = 1.0/(yerr**2) 
	return 0.5*(np.sum((y-model)**2*inv_sigma2))

data = pickle_from_file('data-for-poly-test.pkl')
df = pd.DataFrame(data)

def plot_points():
	fig, ax = plt.subplots()
	ax = sns.regplot('x', 'y', data=df, fit_reg=False)
	ax.errorbar(df['x'],df['y'],df['sigma_y'],fmt='.',color='b')
	plt.show()

x = np.arange(1,10)

def poly_test(dat):
	results = np.zeros(9)
	kek = [[1.0,0.0], [1.0,1.0,0.0], [1.0,1.0,1.0,0.0], [1.0,1.0,1.0,1.0,0.0],[1.0,1.0,1.0,1.0,1.0,0.0], [1.0,1.0,1.0,1.0,1.0,1.0,0.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0], [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0], [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0]]
	for i in x:
		res = op.minimize(neglnL, kek[i-1][:], args=(dat['x'], dat['y'], dat['sigma_y'], i)) 
		results[i-1] = res.fun
		print res.message
	return results	


results = poly_test(df)
fig, ax = plt.subplots()
ax.plot(x, results)
plt.show()

