from astropy.table import Table
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import seaborn as sns

sns.set()
plt.rc('text', usetex=True)


def kde_estimation(table):
	fig, ax = plt.subplots()
	x = table['MBH']
	X = x[:,np.newaxis]
	y = np.zeros(len(table['MBH']))
	Xgrid = np.linspace(0,70,1000)[:, np.newaxis]
	color=cm.rainbow(np.linspace(0,1,7))
	for i in range(1,7):
		kde = KernelDensity(kernel='gaussian', bandwidth=i).fit(X)
		log_dens = kde.score_samples(Xgrid)
		ax.plot(Xgrid, np.exp(log_dens), label='Bandwith = '+str(i), color=color[i])
	plt.scatter(table['MBH'], y , marker='x', color='k') 
	plt.legend()
	plt.title('Gaussian KDE for different bandwiths (in $M_{\odot}$)')
	plt.xlabel('Black Hole mass ($M_{\odot}$)')
	plt.savefig('KDE_BW_plot.svg', dpi=500)
	plt.show()	

def optimal_bandwidth(table, bws, model='gaussian'):
    X = table['MBH']
    N = len(X)
    N_bw = len(bws)
    cv_1 = np.zeros(N_bw)
    X = X.reshape(-1,1)
  
    for i, bw in enumerate(bws):
        print bw
        lnP = 0.0
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            kde = KernelDensity(kernel=model, bandwidth=bw).fit(X_train)
            log_prob = kde.score(X_test)
            lnP += log_prob
        cv_1[i]=lnP/N
    plt.plot(bws, np.exp(cv_1))
    plt.title('Likelihood of different Bandwidths')
    plt.xlabel('Bandwidth')
    plt.ylabel('CV likelihood')
    plt.text(5.8, 0.021, 'Best BW={0:.4f}'.format(bws[np.argmax(cv_1)]))
    plt.savefig('Optimal_BW_plot.svg', dpi=500)
    plt.show()
	
t = Table().read('../../Datasets/joint-bh-mass-table.csv')
kde_estimation(t)
bws = np.linspace(1, 7, 50)
optimal_bandwidth(t, bws)
