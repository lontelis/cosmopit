# cosmopit

cosmological python initial toolkit

It is a toolkit containing routines to perform common and uncommon tasks in cosmology.

Common tasks, that you can perform with this toolkit, are:

1) Simple Statistical tests

2) cosmological coordinate transformations

3) Fitting functions and functionals, with bayesian methods

4) routines to read data from .fits and .txt files

Not-so-common tasks, that you can perform with this toolkit, are:

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

[Functors of Actions example: Actions of Effective Field Theories](https://github.com/lontelis/AofEFT)

[Factor of contaminants Bias D-growth of structure vs redshift](https://github.com/lontelis/FBDz)

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
