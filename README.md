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
<br />
CLASS (software for cosmology https://github.com/lesgourg/class_public/wiki/Installation )  <br />


# INSTALLATION: 
In order to use these libraries one need to download these files: <br />
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
create an additional file using: 

```
touch ~/my_dir/__init__.py 
```

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
please feel free to contact for any queries or bags at: <br />
pntelis -at- apc -point- in2p3 -point- fr
