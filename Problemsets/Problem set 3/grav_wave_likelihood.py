from astropy.table import Table
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from matplotlib.pyplot import cm
from sklearn.model_selection import KFold
import numpy as np
import seaborn as sns

sns.set()
plt.rc('text', usetex=True)

def prob_range(kde, start, end, N=1000):
	step = (end - start) / (N - 1)  # Step size
	x = np.linspace(start, end, N)[:, np.newaxis]  # Generate values in the range
	kde_vals = np.exp(kde.score_samples(x))  # Get PDF values for each x
	probability = np.sum(kde_vals * step)  # Approximate the integral of the PDF
	return probability

def bandwidth(table, bws, model='gaussian'):
	X = table['Mass']
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
	return cv_1

t = Table().read('../../Datasets/pulsar_masses.vot')
y = np.zeros(len(t['Mass']))
bws = np.linspace(0.01, 0.20, 50)

cv_1 = bandwidth(t, bws)
print "Best bandwidth:", bws[np.argmax(cv_1)]

fig, ax = plt.subplots()
bbw = bws[np.argmax(cv_1)]
X = t['Mass']
X = X.reshape(-1,1)
Xgrid = np.linspace(0,3.5,1000)[:, np.newaxis]
kde = KernelDensity(kernel='gaussian', bandwidth=bbw).fit(X)
log_dens = kde.score_samples(Xgrid)
ax.plot(Xgrid, np.exp(log_dens))
ax.set_xlim([0,3.5])
plt.scatter(t['Mass'], y , marker='x', color='k')
plt.title('Kernel Density of Detected Neutron Stars')
plt.xlabel('Neutron star mass ($M_{\odot}$)')
plt.savefig('KDE_NS_Plots.svg', dpi=500)
plt.show()

print "Probability of M > 1.8:", prob_range(kde, 1.8000001, 5)
M1 = prob_range(kde, 1.36, 2.26)
M2 = prob_range(kde, 0.86, 1.36)
print "Probability of M in range [1.36,2.26]:", M1
print "Probability of M in range [0.86,1.36]:", M2
print "Probability of the binary:" , M1*M2

print "Simulating next 5 detections:"
b = kde.sample(10)
for i in range(0,5):
	print "M1:" , b[i][0], ", M2:", b[9-i][0] 
