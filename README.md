# cosmopit
cosmological python initial toolkit

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
```
<br />
and then use each library as following. As an example from the cosmology.py library you can calculate the Comoving Volume distance as: <br />

``` 
distComVol = cosmology.D_V(z=0.5) 
```

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
