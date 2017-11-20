import emcee
import cPickle
import scipy.optimize as op
import numpy as np 
import corner
import seaborn as sns
import matplotlib.pyplot as plt


def pickle_to_file(data, fname):
	fh = open(fname, 'w') 
	cPickle.dump(data, fh) 
	fh.close() 

def pickle_from_file(fname): 
    fh = open(fname, 'r') 
    data = cPickle.load(fh) 
    fh.close() 
    return data

def neglnL(theta, x, y, yerr): 
	a, b = theta 
	model = b * x + a 
	inv_sigma2 = 1.0/(yerr**2) 
	return 0.5*(np.sum((y-model)**2*inv_sigma2))

def lnprior(theta): 
	a, b = theta 
	if -5.0 < a < 5.0 and -10.0 < b < 10.0: 
		return 0.0
	return -np.inf

def lnprob(theta, x, y, yerr): 
	""" 
	The likelihood to include in the MCMC. 
	""" 
	lp = lnprior(theta) 
	if not np.isfinite(lp): 
		return -np.inf 
	return lp - neglnL(theta, x, y, yerr)

d = pickle_from_file('points_example1.pkl')

result = op.minimize(neglnL, [1.0, 0.0], args=(d['x'], d['y'], d['sigma'])) 
a_ml, b_ml = result["x"]
print "Intercept=",a_ml,", slope=",b_ml

p_init = np.array([a_ml,b_ml])
# Set up the properties of the problem.
ndim, nwalkers = 2, 100
# Setup a number of initial positions.
pos = [p_init + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# Create the sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(d['x'], d['y'], d['sigma']))
# Run the process.
sampler.run_mcmc(pos, 500)

samples = sampler.chain[:, 50:, :].reshape((-1, 2))
labels = ['a', 'b'] 
chain = sampler.chain 
for i_dim in range(2): 
	plt.subplot(2,1,i_dim+1) 
	plt.ylabel(labels[i_dim]) 
	for i in range(100): 
		plt.plot(chain[i,:,i_dim],color='black', alpha=0.5)
plt.savefig('Chainsample.svg', bbox_inches='tight', dpi=300)
plt.show()

fig = corner.corner(samples, labels=["$a$", "$b$"],
                    truths=[0.0, 1.3], quantiles=[0.16, 0.84])
plt.savefig('SamplerResult.svg', bbox_inches='tight', dpi=300)
plt.show()


