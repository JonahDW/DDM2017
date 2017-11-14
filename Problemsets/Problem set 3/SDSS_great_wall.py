from astropy.table import Table
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from matplotlib.pyplot import cm
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sns
from astroML.datasets import fetch_great_wall

sns.set()
plt.rc('text', usetex=True)

def bandwidth(X, bws, model='gaussian'):
	N = len(X)
	N_bw = len(bws)
	cv_1 = np.zeros(N_bw)

	for i, bw in enumerate(bws):
	    #print bw
	    lnP = 0.0
	    kf = KFold(n_splits=10)
	    for train_index, test_index in kf.split(X):
	        #print("TRAIN:", train_index, "TEST:", test_index)
	        X_train, X_test = X[train_index], X[test_index]
	        kde = KernelDensity(kernel=model, bandwidth=bw).fit(X_train)
	        log_prob = kde.score(X_test)
	        lnP += log_prob
	    cv_1[i]=lnP/N
	print "Best bandwidth:", bws[np.argmax(cv_1)]
	return cv_1

def grid(Nx, Ny, xmin, xmax, ymin, ymax):
	xgrid = np.linspace(xmin, xmax, Nx)
	ygrid = np.linspace(ymin, ymax, Ny)
	mesh = np.meshgrid(xgrid, ygrid)
	tmp = map(np.ravel, mesh)
	Xgrid = np.vstack(tmp).T
	return Xgrid

def plot_kernels(X, Xgrid):
	kernels = ['gaussian','tophat','exponential','epanechnikov']

	fig, axs = plt.subplots(2,2, figsize=(10,10), sharex=True, sharey=True)
	axs = axs.ravel()

	for i, kernel in enumerate(kernels):
		kde = KernelDensity(kernel=kernel, bandwidth=5).fit(X)
		log_dens = kde.score_samples(Xgrid)
		dens1 = X.shape[0]*np.exp(log_dens).reshape((Ny,Nx))
		im = axs[i-1].contourf(dens1)
		axs[i-1].set_title("Kernel: "+str(kernel))

	fig.colorbar(im, ax=axs.tolist())
	plt.suptitle('Density plots for The Great Wall')
	plt.savefig('KDE_GW_cont.svg', dpi=500)
	#plt.tight_layout()
	plt.show()

X = fetch_great_wall()
Nx = 50
Ny = 125
Xgrid = grid(Nx, Ny, -375, -175, -300, 200)
bws = np.linspace(7.9, 8.1, 50)

#cv = bandwidth(X, bws)
#bbw = bws[np.argmax(cv)]

kde = KernelDensity(kernel='gaussian', bandwidth=7.99).fit(X)
log_dens = kde.score_samples(Xgrid)
dens1 = X.shape[0]*np.exp(log_dens).reshape((Ny,Nx))

plt.contourf(dens1)
plt.title('Kernel Density of the Great Wall (Bandwith='+str(7.99)+')')
plt.colorbar()
plt.savefig('KDE_GW_BW.svg', dpi=500)
plt.show()