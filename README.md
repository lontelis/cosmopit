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
In order to use these libraries one need to download these files: <br />
filename.py <br />
such as: <br />
cosmology.py <br />
pk_Class.py <br />
SplineFitting.py <br />
fitting.py <br />
galtools.py <br />
<br />
on your directory: <br />
```
cd ~/my_dir 
```
<br />
and create the magic filename: <br />
__init__.py <br />

# USE example:  
import each file like: <br />
``` 
import cosmology 
```
<br />
and then use one library to calculate the Comoving Volume distance as: <br />

``` 
distComVol = cosmology.D_V(z=0.5) 
```

# QUERIES:
please feel free to contact for any queries or bugs at: <br />
pntelis -at- cppm -point- in2p3 -point- fr
