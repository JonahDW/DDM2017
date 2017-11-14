from astropy.table import Table
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from matplotlib.pyplot import cm
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sns
import cPickle

sns.set()
plt.rc('text', usetex=True)

def pickle_from_file(fname): 
    fh = open(fname, 'r') 
    data = cPickle.load(fh) 
    fh.close() 
    return data

def bandwidth(X, bws, model='gaussian'):
	N = len(X)
	N_bw = len(bws)
	cv_1 = np.zeros(N_bw)
	X = X.reshape(-1,1)

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


t = pickle_from_file('../../Datasets/mysterious-peaks.pkl')
y = np.zeros(len(t))
bws = np.linspace(0.01, 5, 50)
#cv_1 = bandwidth(t, bws)

fig, ax = plt.subplots()
bbw = 1.83673469388
X = t
X = X.reshape(-1,1)
Xgrid = np.linspace(-15,15,1000)[:, np.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth=bbw).fit(X)
log_dens = kde.score_samples(Xgrid)
ax.plot(Xgrid, np.exp(log_dens))
ax.set_xlim([-15,15])
plt.scatter(t, y , marker='x', color='k')
plt.title('Kernel Density of Mysterious Peaks (2 Peaks)')
#plt.xlabel('Neutron star mass ($M_{\odot}$)')
plt.savefig('KDE_MP_Plots.svg', dpi=500)
plt.show()