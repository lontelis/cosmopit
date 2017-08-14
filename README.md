# cosmopit
cosmological python initial tools

# Prerequist
python 2.7+ () \\
numpy \\
pylab \\
scipy  \\
pyfits \\
iminuit (https://pypi.python.org/pypi/iminuit) \\
emcee   (http://dan.iel.fm/emcee/current/) \\

CLASS (software for cosmology https://github.com/lesgourg/class_public/wiki/Installation ) \\


# INSTALLATION: 
In order to use these libraries one need to download these files: \\
cosmology.py \\
pk_Class.py \\
SplineFitting.py \\
fitting.py \\
galtools.py \\

on your directory: \\
~/my_dir \\

create a file using: \\
touch ~/my_dir/__init__.py \\

# USE example:
and import each file like: \\

import cosmology.py \\

and then use the function to calculate the Comoving Volume distance as: \\

distComVol = cosmology.D_V(z=0.5) \\


