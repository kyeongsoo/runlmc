#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('env', 'OMP_NUM_THREADS=1')

import os, sys
if os.path.basename(os.getcwd()) != 'runlmc':
    os.chdir('..')
sys.path.append('benchmarks/benchlib')

from runlmc.models.interpolated_llgp import InterpolatedLLGP
from runlmc.lmc.functional_kernel import FunctionalKernel
from runlmc.kern.rbf import RBF
from runlmc.models.optimization import AdaDelta

import numpy as np

# get_ipython().run_line_magic('matplotlib', 'inline')
# import matplotlib.pyplot as pltv
import matplotlib.pyplot as plt


np.random.seed(1234)


# In[2]:


n_per_output = [65, 100]
xss = list(map(np.random.rand, n_per_output))
nout = len(n_per_output)
yss = [np.sin(2 * np.pi * xs + i * 2 * np.pi / nout)
       + np.random.randn(len(xs)) * (i + 1) * 0.1 / nout
       for i, xs in enumerate(xss)]
ks = [RBF(name='rbf{}'.format(i)) for i in range(nout)]
ranks = [1, 1]
fk = FunctionalKernel(D=len(xss), lmc_kernels=ks, lmc_ranks=ranks)


# In[3]:


def print_diagram(lmc):
    plot_xs =  np.arange(0, 1, 0.01)
    ys, var = lmc.predict([plot_xs for _ in range(nout)])
    for i, (y, v, xs, ys) in enumerate(zip(ys, var, xss, yss)):
        sd = np.sqrt(v)
        order = xs.argsort()
        plt.scatter(xs[order], ys[order])
        plt.title('output {} (95%)'.format(i))
        plt.plot(plot_xs, y)
        plt.plot(plot_xs, y + 2 * sd, ls='--', c='g')
        plt.plot(plot_xs, y - 2 * sd, ls='--', c='g')
        plt.show()


# In[4]:


# Unoptimized
lmc = InterpolatedLLGP(xss, yss, functional_kernel=fk)
print_diagram(lmc)


# In[5]:


lmc.optimize(optimizer=AdaDelta(verbosity=10))
#optimized
print(lmc)
print(lmc.kern.noise)
print_diagram(lmc)


# In[6]:


import GPy

rbfs = [GPy.kern.RBF(1) for _ in range(nout)]
# not exactly the same since mine is rank-1 only for now
# This is why we need as many kernels as outputs, because we'd be rank-deficient o/w
k = GPy.util.multioutput.LCM(input_dim=1,num_outputs=nout,kernels_list=rbfs)
xss_reshaped = [xs.reshape(-1, 1) for xs in xss]
yss_reshaped = [ys.reshape(-1, 1) for ys in yss]
m = GPy.models.GPCoregionalizedRegression(
    xss_reshaped, yss_reshaped, kernel=k)
m.optimize()
print(m)


# In[7]:


# Plotting code adapted from GPy coregionalization tutorial
# Also 95% confidence

data_rows = np.add.accumulate(n_per_output)
data_rows = np.insert(data_rows, 0, 0)

for i in range(nout):
    m.plot(
        plot_limits=(0, 1),
        fixed_inputs=[(1,i)],
        which_data_rows=slice(data_rows[i],data_rows[i + 1]))


# In[8]:


# Adding a prior
from runlmc.parameterization.priors import InverseGamma, Gaussian, HalfLaplace

ig = InverseGamma(0.5, 0.5)
lmc.kern.rbf0.inv_lengthscale.set_prior(ig)
lmc.kern.rbf1.inv_lengthscale.set_prior(ig)

n = Gaussian(0, 1)
lmc.kern.a0.set_prior(n)
lmc.kern.a1.set_prior(n)

h = HalfLaplace(1)
lmc.kern.kappa0.set_prior(h)
lmc.kern.kappa1.set_prior(h)

lmc.optimize()

print(lmc)
print(lmc.kern.kappa0)


# In[9]:


# Multilevel prior
from runlmc.parameterization.param import Param

# A param is anything that is modifiable during the optimization
# We add a param shape such that
# shape ~ IG(0.5, 0.5)
# rbf*.inv_lengthscale ~ IG(0.5, shape)

ig = InverseGamma(0.5, 0.5)
initial_value = 1
shape = Param('shape', initial_value)
lmc.link_parameter(shape) # wire the parameter into the model (otherwise it won't get updated)
shape.set_prior(ig)

ig2 = InverseGamma(0.5, shape)
for il in [lmc.kern.rbf0.inv_lengthscale, lmc.kern.rbf1.inv_lengthscale]:
    il.set_prior(ig2)
    
lmc.optimize(optimizer=AdaDelta())
print(lmc)
print(lmc.kern.kappa0)


# In[ ]:




