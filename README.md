# cosmopit

### cosmopit

Cosmopit (Cosmological python initial toolkit) is a package of cosmology built on basic numerical libraries of python such as Numpy/SciPy/Matplolib. Capabilities include common and not-so-common capabilities. 

Common capabilities include: various cosmological distances, and volumes calculations, transformation of cosmological coordinate systems, simple statistic quantities and also calculations of covariance matrices, fitting routines and bayesian inferences using corner plots, routines to read and write date in .fits and .txt files, example of calculating Monte Carlo Markov Chains for fittings and corner plot presentation tutorial, and a tutorial dedicated for transformation of cosmological coordinates.

Not-so common capabilities include: calculation of power spectra and correlation functions via the class software, calculation of number counts, fractal dimension quantities, homogeneity scale calculation routines, routines calculating statistics of simualated, reading and writing routines for cute, astropy, pymangle, 
example of Baryon Acoustic Oscilation (BAO) detection and extraction feature from data, example of fitting the SuperNovae (SN) curve to extra simple magnitude parameters.

https://github.com/lontelis/cosmopit


cosmological python initial toolkit

It is a toolkit containing routines to perform common and uncommon tasks in cosmology.

Common tasks, that you can perform with this toolkit, are:

1) Simple Statistical tests

2) cosmological coordinate transformations

3) Fitting functions and functionals, with bayesian methods

4) routines to read data from .fits and .txt files

Unommon tasks, that you can perform with this toolkit, are:

1) Calculation of theoretical predictions of 2point correlation function, number counts, fractal dimension
   using the CLASS software.

2) Calculation of estimators of 2point correlation function, number counts, fractal dimension,
   using the cute software

The cosmology used assumes a homogeneous and isotropic universe within the ΛCDM model.



# INSTALLATION: 
In order to use these libraries one need to download these files in their directory: <br />
<br />
Simply do on terminal:
```
git clone https://github.com/lontelis/cosmopit.git
```
or 
```
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps cosmopit
```
<br />

# Prerequist
python 2.7+ <br />
numpy <br />
pylab <br />
scipy <br />
pyfits <br />
iminuit (https://pypi.python.org/pypi/iminuit) <br />
emcee   (http://dan.iel.fm/emcee/current/) <br />
pymc    (https://pymc-devs.github.io/pymc/) <br />   
<br />
CLASS (software for cosmology https://github.com/lesgourg/class_public/wiki/Installation )  <br />
<br />
To install CLASS, just do:
```
git clone https://github.com/lesgourg/class_public.git class
```
<br />

# USE example:  
import each file as: <br />
``` 
from cosmopit import cosmology 
from cosmopit import numMath
from cosmopit import galtools
```
<br />
and then use each library as following. As an example from the cosmology.py library you can calculate the Comoving Volume distance as: <br />

``` 
distComVol = cosmology.D_V(z=0.5) 
```

Other Examples

[Tutorial of coordinate transformation applied in cosmology](https://github.com/lontelis/Tutorial-of-coordinate-transformation-applied-in-cosmology)

[Baryon Acoustic Oscillation (BAO) detection jupyter notebook example](https://github.com/lontelis/Extract-Cosmology-with-the-BAO-peak-position/blob/master/analyse_DR12_BAO.ipynb)

[Supernovae (SN) fitting curve jupyter notebook example ](https://github.com/lontelis/ANALYSE-SN-magnitude-redshift-curve/blob/master/analyse_SN.ipynb)

[MCMC python example](https://github.com/lontelis/MCMC-posterior-example)

# Citation
If you use this code please cite at least the following paper: <br />
P.Ntelis et al 2018 https://arxiv.org/abs/1810.09362 <br />
<br />

Citable papers which use partially the aforementioned code package 
(and which can be also be cited for the same reason)
are: <br />

The scale of cosmic homogeneity as a standard ruler <br />
P.Ntelis et al 2018 https://arxiv.org/abs/1810.09362  <br /> <br />
Exploring cosmic homogeneity with the BOSS DR12 galaxy sample <br /> 
P.Ntelis et al 2017 https://arxiv.org/abs/1702.02159 <br /> <br />
A 14 h−3 Gpc3 study of cosmic homogeneity using BOSS DR12 quasar sample <br />
P.Laurent, ..., P. Ntelis, et al. https://arxiv.org/abs/1602.09010

# QUERIES:
please feel free to contact for any queries or bugs at: <br />
ntelis.pierros -at- gmail -point- com
