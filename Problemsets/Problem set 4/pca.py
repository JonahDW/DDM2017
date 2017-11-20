from astropy.table import Table
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

t = Table.read('../../Datasets/x-vs-y-for-PCA.csv')

x = t['x']
y = t['y']

def standardize(x):
	x_mean = np.mean(x)
	x_std = np.std(x)
	x_new = (x-x_mean)/x_std
	return x_new

def unstandardize(X, x):
	x_mean = np.mean(x)
	x_std = np.std(x)
	X_back = X*x_std+x_mean
	return X_back

x_new, y_new = standardize(x), standardize(y)

c_x = np.cov(x_new, y_new)
val, vec = np.linalg.eig(c_x)

def eigen_comps(x_new, y_new, eigenvec):
	pcs = np.empty((2,len(x_new)))
	for i in range(len(x_new)):
		pcs[:, i] = np.matmul(eigenvec.T, np.array([x_new[i], y_new[i]]).T)
	return pcs


def plot_eigenvec(x_new, y_new, eigenvec):
	a1 = eigenvec[0,0]/eigenvec[0,1]
	a2 = eigenvec[1,0]/eigenvec[1,1]
	xx = np.linspace(-10,10,50)
	plt.scatter(x,y)
	plt.plot(a1*xx, xx)
	plt.plot(a2*xx, xx)
	plt.xlim(-2,5)
	plt.ylim(-4,12)

def project_back(pcs, eigenvec):
	X = pcs*eigenvec[0]
	Y = pcs*eigenvec[1]
	x_back = unstandardize(X, x)
	y_back = unstandardize(Y, y)
	return x_back, y_back


#plt.scatter(x,y)
#plt.scatter(x_new,y_new)
#plt.show()

pcs = eigen_comps(x_new, y_new, vec[0,:])
x_back, y_back = project_back(pcs[0,:], vec[0,:])

#plt.scatter(x,y)
plt.scatter(pcs[0,:]*vec[0,0], pcs[1,:]*vec[0,1], color='g')
plt.scatter(x_back, y_back, color='r')
plt.scatter(pcs[0,:], pcs[1,:])
plt.show()