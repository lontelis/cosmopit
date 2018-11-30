# cosmopit
cosmological python initial toolkit

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


# INSTALLATION: 
In order to use these libraries one need to download these files in their directory: <br />
Rename the package cosmopit <br />

# USE example:  
import each file like: <br />
``` 
from cosmopit import cosmology 
from cosmopit import numMath
```
<br />
and then use each library to calculate the Comoving Volume distance as: <br />

``` 
distComVol = cosmology.D_V(z=0.5) 
```

# Citation
If you use this code please cite at least the following papers: <br />
P.Ntelis et al 2018 https://arxiv.org/abs/1810.09362 <br />
<br />

Papers that use parts of the aforementioned code package are: <br />

The scale of cosmic homogeneity as a standard ruler <br />
P.Ntelis et al 2018 https://arxiv.org/abs/1810.09362  <br /> <br />
Exploring cosmic homogeneity with the BOSS DR12 galaxy sample <br /> 
P.Ntelis et al 2017 https://arxiv.org/abs/1702.02159 <br /> <br />
A 14 h−3 Gpc3 study of cosmic homogeneity using BOSS DR12 quasar sample <br />
P.Laurent, ..., P. Ntelis, et al. https://arxiv.org/abs/1602.09010

# QUERIES:
please feel free to contact for any queries or bugs at: <br />
pntelis -at- cppm -point- in2p3 -point- fr
