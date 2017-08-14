import cosmology,pk_Class
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import pyfits
import string
import random
import subprocess
import os
import glob
from scipy import integrate
from scipy import interpolate
from scipy import linalg
import scipy
import emcee
import scipy.optimize as opt
import numMath,SplineFitting
import fitting
import healpy as hp
from astropy import coordinates
import astropy.io.fits as afits
# Taken from J.C.Hamilton and remodified by P.Ntelis June 2014

#######
#
# changed function:
# read_mocks
# and homogeneity_many_pairs
# 'in order to acount for bias'
#
#######

'''
host = os.environ['HOST']

if host[0:6] == 'dapint':
    print "enter host if dapint"
    docfmpi = "/home/usr202/mnt/burtin/Cosmo/analyse/python_from_jch/Homogeneity/docfmpi.py"
    mpirun = "mpirun"
    
elif host == 'MacBook-Pro-de-Jean-Christophe.local' or host == 'apcmc191.in2p3.fr' :
    print "enter host if JCH_.local"
    docfmpi = "/Users/hamilton/idl/pro/SDSS-APC/python/Homogeneity/docfmpi.py"
    mpirun = "openmpirun"
    
if host == 'apcmcqubic' :
    print "enter host if apcmcqubic"
    docfmpi = "/Users/hamilton/idl/pro/SDSS-APC/python/Homogeneity/docfmpi.py"
    mpirun = "/opt/local/bin/openmpirun" 
else :
    message = "********* in galtools.py, unknown host : " + host + "  *******************"
    print(message)
'''

def meancut(array,nsig=3,niter=3):
    ''' computes the mean and std removing those that are 3sigma away from the mean, x3'''
    ''' Should we keep only those that are inside the 1 and 3 Quartile? and make the average? '''
    thearray=array
    for i in np.arange(niter):
        m=np.mean(thearray)
        s=np.std(thearray)
        w=np.where(np.abs(thearray-m) <= nsig*s)
        thearray=thearray[w[0]]
    return(m,s,thearray)

def cov2cor(mat):
    cor=np.zeros((mat.shape[0],mat.shape[1]))
    for i in np.arange(mat.shape[0]):
        for j in np.arange(mat.shape[1]):
            cor[i,j]=mat[i,j]/np.sqrt(mat[i,i]*mat[j,j])
    return(cor)

def pick_indices(done,nfiles):
    # randomly!!
    okrand=False
    icount=0
    while okrand is False:
        thenumrand=int(np.floor(random.random()*nfiles))
        if thenumrand not in done:
            okrand=True
            numrand=thenumrand
            #print('OK for ',numrand)
        else:
            #print('    ',thenumrand,' was already in the list')
            okrand=False
            icount=icount+1
        if icount >= nfiles:
            print('Error in pick_indices: Cannot find enough free indices')
            stop
    return numrand

def random_string(nchars):
    lst = [random.choice(string.ascii_letters + string.digits) for n in xrange(nchars)]
    str = "".join(lst)
    return(str)
## Transformations



def xyz2rthph(x,y,z):
    rho   = np.sqrt(x*x+y*y+z*z)
    theta = np.arccos(z/rho) 
    phi   = np.arctan2(y,x)  
    'correction to go to [0,2pi] according to https://en.wikipedia.org/wiki/Atan2'
    if type(phi) == np.ndarray:
        phi[np.where(phi<0.)] += 2*np.pi
        ##### PL suggestion phi   = np.arccos(x/(rho*np.sin(theta)))
    else: 
        if phi<0.: phi += 2*np.pi
        else: pass 

    return rho,theta,phi

def rthph2zdecra(rho,theta,phi,params=[0.3,0.7,-1.0,0.0]):
    #h = params[4]
    redshift = cosmology.get_dist(rho,params=params,z2radial=True)
    dec = 90. - 180./np.pi*theta
    ra  = 180./np.pi*phi
    return redshift,dec,ra

def xyz2zdecra(x,y,z,params=[0.3,0.7,-1.0,0.0]):
    ''' need to rotate to theta and phi so that rthph2zdecra go to right angles'''
    rho,theta,phi   = xyz2rthph(x,y,z)
    redshift,dec,ra = rthph2zdecra(rho,theta,phi,params=params) 
    return redshift,dec,ra

def zdecra2rthph(z_array,dec_array,ra_array,params=[0.3,0.7,-1.0,0.0],dist_type='proper'):
    #h = params[4]
    r_array=cosmology.get_dist(z_array,type=dist_type,params=params) # in Mpc/h
    th_array=(90.-dec_array)*np.pi/180.
    ph_array=ra_array*np.pi/180.
    return(r_array,th_array,ph_array)

def rthph2xyz(r,th,ph):
    """
    for r, theta (rad) and phi (rad), returns the x,y,z in Euclidean coordinates
    """
    x=r*np.sin(th)*np.cos(ph)
    y=r*np.sin(th)*np.sin(ph)
    z=r*np.cos(th)
    return(x,y,z)   

def zdecra2xyz(redshift,dec,ra,params=[0.3,0.7,-1.0,0.0],dist_type='proper'):
    r,th,ph= zdecra2rthph(redshift,dec,ra,params=params,dist_type=dist_type)
    x,y,z  = rthph2xyz(r,th,ph)
    return x,y,z

    # TESTS
def test_xyz2zdecra_zdecra2xyz(x=0, y=0, z=1, rtol=1e-5):
    red, dec, ra = xyz2zdecra(x, y, z)
    x_, y_, z_ = zdecra2xyz(red, dec, ra)
    print x_, y_, z_
    print np.allclose( x , x_, rtol=rtol) , np.allclose(y , y_,rtol=rtol) , np.allclose( z , z_,rtol=rtol)
    return x == x_ and y == y_ and z == z_

def test_zdecra2rthph_rthph2zdecra(z=0.5,dec=20.,ra=150.,rtol=1e-4):
    r,th,ph = zdecra2rthph(z,dec,ra)
    z_,dec_,ra_ = rthph2zdecra(r,th,ph)
    print z_,dec_,ra_
    print np.allclose(z,z_ , rtol=rtol), np.allclose(dec,dec_, rtol=rtol) , np.allclose(ra,ra_, rtol=rtol)
    print 'interpolation z=z(r) effect'
    return np.allclose(z,z_) and np.allclose(dec,dec_) and np.allclose(ra,ra_)

def test_xyz2rthph_rthph2xyz(x=0.,y=0.,z=1.):
    r,th,ph  = xyz2rthph(x,y,z)
    x_,y_,z_ = rthph2xyz(r,th,ph)
    print x_,y_,z_
    print x==x_ , y==y_ , z==z_
    return x==x_ and y==y_ and z==z_ 

def test_rthph2xyz_xyz2rthph(r=1.,th=0.,ph=0.):
    x,y,z = rthph2xyz(r,th,ph)
    r_,th_,ph_ = xyz2rthph(x,y,z)
    print r_,th_,ph_
    print r==r_,th==th_,ph==ph_
    return r==r_ and th==th_ and ph==ph_

## Transformations

### PAIRS NIkhil

def rthphw2fits(fitsname,r,th,ph,w=None):
    """
    for r, theta (rad) and phi (rad) and optionnaly w (weights), writes them in euclidean coordinates into a fits file with name fitsname
    """
    x,y,z=rthph2xyz(r,th,ph)
    nb=x.size
    if w is None: w=np.ones(nb)
    
    #weights need to be normalized to an average of 1
    w=w/np.mean(w)

    #Writing Fits file
    col0=afits.Column(name='x',format='E',array=x)
    col1=afits.Column(name='y',format='E',array=y)
    col2=afits.Column(name='z',format='E',array=z)
    col3=afits.Column(name='w',format='E',array=w)
    cols=afits.ColDefs([col0,col1,col2,col3])
    tbhdu=afits.new_table(cols) #BinTableHDU.from_columns(cols)
    tbhdu.writeto(fitsname,clobber=True)
    return(x,y,z,w)


def run_kdtree(datafile1,datafile2,binsfile,resfile,counter="euclidean",nproc=None):
    print "  entered galtools.run_kdtree"
    if nproc is None:
        subprocess.call(["python",
                         docfmpi,
                         "--counter="+counter,
                         "-b",binsfile,
                         "-o",resfile,
                         "-w",
                         datafile1,
                         datafile2])
    else:
        subprocess.call([mpirun,
                         "-np",str(nproc),
                         "python",
                         docfmpi,
                         "--counter="+counter,
                         "-b",binsfile,
                         "-o",resfile,
                         "-w",
                         datafile1,
                         datafile2])

def read_pairs(file,normalize=True):
    bla=np.loadtxt(file,skiprows=1)
    r=bla[:,0]
    dd=bla[:,1]
    rr=bla[:,2]
    dr=bla[:,3]
    f=open(file)
    a=f.readline()
    ng=float(a.split("=")[1].split(" ")[0])
    nr=float(a.split("=")[2].split(" ")[0])
    f.close()
    
    if normalize is True:
        dd=dd/(ng*(ng-1.)/2.) #=dd/(ng*(ng-1)/2) (on nersc when I use the /2 I got false correlation function)
        rr=rr/(nr*(nr-1.)/2.) #=rr/(nr*(nr-1)/2) (here I need to use it because I devide by 2 when I compute the pairs)
        dr=dr/(ng*nr)                           #(here if I dont use /2 I get wrong (dd-2.*dr+rr)/rr)
    
    return r,dd,rr,dr,ng,nr

def read_pairs_combine_regions(file1,file2,normalize=True):
    r1,dd1,rr1,dr1,ng1,nr1=read_pairs(file1,normalize=False)
    r2,dd2,rr2,dr2,ng2,nr2=read_pairs(file2,normalize=False)
    ng=ng1+ng2
    nr=nr1+nr2
    dd=dd1+dd2
    rr=rr1+rr2
    dr=dr1+dr2

    if normalize is True:
        dd=dd/(ng*(ng-1.)/2.) #=dd/(ng*(ng-1)/2) (on nersc when I use the /2 I got false correlation function)
        rr=rr/(nr*(nr-1.)/2.) #=rr/(nr*(nr-1)/2)
        dr=dr/(ng*nr)

    return r1,dd,rr,dr,ng,nr

    """#!#
def read_many_pairs(files):
    r,dd,rr,dr,ng,nr=read_pairs(files[0])
    nsim=np.size(files)
    nbins=np.size(r)

    all_ls=np.zeros((nsim,nbins))
    i=0
    for file in files:
        r,dd,rr,dr,ng,nr=read_pairs(file)
        ls=(dd-2*dr+rr)/rr
        all_ls[i,:]=ls
        i=i+1

    meanls,sigls,covmat,cormat=average_realisations(all_ls)
    return(r,meanls,sigls,covmat,cormat)

def read_many_pairs_combine(filesS,filesN):
    r,dd,rr,dr,ng,nr=read_pairs(filesS[0])
    nsim=np.size(filesN)
    nbins=np.size(r)
    
    all_ls=np.zeros((nsim,nbins))
    i=0
    for fileN,fileS in zip(filesN,filesN):
        r,dd,rr,dr,ng,nr=read_pairs_combine_regions(fileN,fileS)
        ls=(dd-2*dr+rr)/rr
        all_ls[i,:]=ls
        i=i+1

    meanls,sigls,covmat,cormat=average_realisations(all_ls)
    return(r,meanls,sigls,covmat,cormat)
    #!#"""

def get_pairs(rdata,thdata,phdata,rrandom,thrandom,phrandom,rmin,rmax,nbins,wdata=None,wrandom=None,counter="euclidean",nproc=None,log=None):
    """
    r,dd,rr,dr=get_pairs(rdata,thdata,phidata,rrandom,thrandom,phirandom,rmin,rmax,nbins,wdata=None,wrandom=None,nproc=None,log=None)
    """
    # Need a random string for temporary files
    print "   entered galtools.get_pairs"
    rndstr=random_string(10)

    # Prepare filenames
    datafile="/tmp/data_"+rndstr+".fits"
    randomfile="/tmp/random_"+rndstr+".fits"
    binsfile="/tmp/bins_"+rndstr+".txt"
    ddfile="/tmp/dd_"+rndstr+".txt"
    rrfile="/tmp/rr_"+rndstr+".txt"
    drfile="/tmp/dr_"+rndstr+".txt"

    # write fits files with data and randoms
    xd,yd,zd,wd=rthphw2fits(datafile,rdata,thdata,phdata,w=wdata)
    xr,yr,zr,wr=rthphw2fits(randomfile,rrandom,thrandom,phrandom,w=wrandom)
    # write bins file
    if log is None:
        edges=np.linspace(rmin,rmax,nbins+1)
    else:
        edges1=np.linspace(0.,rmin,1)
        edges2=10.**(np.linspace(np.log10(rmin),np.log10(rmax),nbins+1))
        edges = np.concatenate((edges1,edges2))
        print edges1
    print np.min(edges),np.max(edges)

    outfile=open(binsfile,'w')
    for x in edges:
        outfile.write("%s\n" % x)

    outfile.close()
    # do DD
    print('       - Doing DD : '+str(xd.size)+' elements')
    run_kdtree(datafile,datafile,binsfile,ddfile,counter=counter,nproc=nproc)
    alldd=np.loadtxt(ddfile,skiprows=1)
    # do RR
    print('       - Doing RR : '+str(xr.size)+' elements')
    run_kdtree(randomfile,randomfile,binsfile,rrfile,counter=counter,nproc=nproc)
    allrr=np.loadtxt(rrfile,skiprows=1)
    # do DR
    print('       - Doing DR : '+str(xd.size)+'x'+str(xr.size)+' pairs')
    run_kdtree(datafile,randomfile,binsfile,drfile,counter=counter,nproc=nproc)
    alldr=np.loadtxt(drfile,skiprows=1)

    dd=alldd[:,2]
    rr=allrr[:,2]
    dr=alldr[:,2]

    # correct DD and RR for double counting (I just add "." for correction maybe?)
    dd=dd/2.
    rr=rr/2.

    r=(alldd[:,0]+alldd[:,1])/2
    subprocess.call(["rm","-f",datafile,randomfile,binsfile,ddfile,rrfile,drfile])
    return(r,dd,rr,dr)


def paircount_data_random(datara,datadec,dataz,randomra,randomdec,randomz,cosmo_model,rmin,rmax,nbins,counter="euclidean",nproc=None,log=None,file=None,wdata=None,wrandom=None):
    """
    r,dd,rr,dr=paircount_data_random(datara,datadec,dataz,randomra,randomdec,randomz,cosmo,rmin,rmax,nbins,log=None,file=None,wdata=None,wrandom=None)
    Returns the R,DD,RR,DR for a given set of data and random and a given cosmology
    """

    # calculate proper distance for each object (data and random)
    print "   entered galtools.paircount_data_random"
    pi=np.pi
    params=cosmo_model[0:4]
    #h=cosmo_model[4]
    rdata,thdata,phdata = zdecra2rthph(dataz,datadec,datara,params=params,dist_type='prop') # in Mpc/h
    rrandom,thrandom,phrandom = zdecra2rthph(randomz,randomdec,randomra,params=params,dist_type='prop') # in Mpc/h
    # Count pairs
    r,dd,rr,dr=get_pairs(rdata,thdata,phdata,rrandom,thrandom,phrandom,rmin,rmax,nbins,counter=counter,nproc=nproc,log=log,wdata=wdata,wrandom=wrandom)

    # Write to file if required
    if file is not None:
        outfile=open(file,'w')
        if wdata==None: 
            Norm_gala = np.size(datara)
        else: 
            Norm_gala = np.sum(wdata)
        if wrandom==None: 
            Norm_rand = np.size(randomra)
        else: 
            Norm_rand = np.sum(wrandom)

        outfile.write("Ng=%s Nr=%s \n" % (Norm_gala, Norm_rand))
        print 'Norm_gala        Norm_rand'
        print np.sum(wdata),np.sum(wrandom)
        for xr,xdd,xrr,xdr in zip(r,dd,rr,dr):
            outfile.write("%s %s %s %s\n" % (xr,xdd,xrr,xdr))

    outfile.close()

    # return result
    return(r,dd,rr,dr)

### PAIRS NIkhil

############### Observables Xi(r) N(r) and D2(r) ##########################

def xi(dd,rr,dr,who_xi='ph',corr_lss=False):
    if who_xi=='ph':
        res = (dd/rr - 1.)
    elif who_xi=='h':
        res = (dd*rr)/(dr*dr) -1.
    elif who_xi=='dp':
        res = dd/dr - 1.
    elif who_xi=='ls':
        res = (dd-2.*dr+rr)/rr

    # if True: puts xi(r_lss)=0 where cute gives rr(r_lss)=0
    if corr_lss: res[np.where(np.isnan(res))]=np.zeros( len(np.where(np.isnan(res))[0]) ) 

    return res

def N_JM(r,dd,rr,dr,bias):
    """
    if bias  = 1 gives the N(r)
    if bias != 1 gives the unbiased N(r)
    """
    #N_gal = r*0.
    xi_ls_arithm = dd-2.*dr+rr
    
    #N_arith = integrate.cumtrapz(xi_ls_arithm,x=r,initial=xi_ls_arithm[0]) # initial=xi_ls_arithm[0]
    #N_paran = integrate.cumtrapz(rr,x=r,initial=rr[0])                     # initial=rr[0]
    
    N_arith = np.cumsum(xi_ls_arithm)
    N_paran = np.cumsum(rr)           

    N_gal = 1. + N_arith/N_paran
    #give N_DM ~ N_all_matter
    N_MM = ( (N_gal - 1)/(bias**2) ) + 1.

    return N_MM

def N_JC(r,dd,rr,dr,bias,who_xi='ls'):
    N_biased = r*0.
    xi_ls = xi(dd,rr,dr,who_xi=who_xi)
    
    # correction of integration
    #integral_correction = xi_ls[0] * r[0] #*2 # r[0]=(1.-0)/2. 

    #xi_ls_r2 = xi_ls[1:] * (r[1:]**2) 
    xi_ls_r2 = xi_ls * r*r
    
    #Integration = integral_correction + integrate.cumtrapz(xi_ls_r2,x=r[1:],initial=xi_ls_r2[0]) #  initial=xi_ls_r2[0]
    #Integration = integral_correction + integrate.cumtrapz(xi_ls_r2,x=r,initial=xi_ls_r2[0]) #  initial=xi_ls_r2[0]
    Integration = integrate.cumtrapz(xi_ls_r2,x=r,initial=xi_ls_r2[0]) #  initial=xi_ls_r2[0]
    
    #N_biased = 1 + (3./ (r**3) )*np.concatenate((np.array([integral_correction]),Integration))
    N_biased = 1 + (3./ (r**3) )*Integration

    N_unbiased = ( (N_biased - 1)/(bias**2) ) + 1

    return N_unbiased

def N_str_JC(r,dd,rr,bias):
    y0 = dd[0]/rr[0]
    y = dd/rr
    #N_gal = integrate.cumtrapz(y,x=r,initial=y0)
    N_gal = np.cumsum(y)
    N_MM = ( (N_gal-1)/(bias**2) ) + 1
    return N_MM
 
def N_str(r,dd,rr,bias):
    #N_gal = integrate.cumtrapz(dd,x=r,initial=dd[0])/integrate.cumtrapz(rr,x=r,initial=rr[0]) # initial=dd[0] initial=rr[0]
    N_gal = np.cumsum(dd)/np.cumsum(rr)
    N_MM = ( (N_gal - 1)/(bias**2) ) + 1
    return N_MM

def d2_nr_compute(r,dd,rr,dr,bias,who_Nest='JC'):
    """
    uses all possible estimators of N(r)
    """
    dlogr=np.gradient(np.log(r))
    #dlogr=np.diff(np.log(r))
    if who_Nest=='JC':
        nr=N_JC(r,dd,rr,dr,bias)
        stop
    elif who_Nest == 'JM':
        nr=N_JM(r,dd,rr,dr,bias)
    elif who_Nest == 'str': 
        nr=N_str(r,dd,rr,bias)
    elif who_Nest == 'strJC':
        nr=N_str_JC(r,dd,rr,bias)
    d2=np.gradient(np.log(nr),dlogr)+3
    #d2=np.diff(np.log(nr))/dlogr+3
    return(d2,nr)

def N_of_xi(r,xi):
    N_gal = r*.0
    xi_r2 = xi * r * r 
    N_gal = 1. + (3. / (r**3.) )*(integrate.cumtrapz(xi_r2,x=r,initial=xi_r2[0])) # initial=xi_r2[0]
    return(N_gal)

def d2_of_N(r,nr,mu=None,gradient=True,which_gradient=False):
    if gradient:
        dlogr = np.gradient(np.log(r))
        d2_res = np.gradient(np.log(nr),dlogr) + 3.
    elif np.shape(np.shape(nr))==(2,): # NEW FEATURE NEED TO BE INTEGRATED TO ALL FUNCTIONS
        if which_gradient:
            d2_res = np.gradient( np.log(nr) , np.gradient(np.log(r))[:,None] , axis=0) + 3.
        else:
            d2_res = np.gradient( np.gradient(np.log(nr), np.gradient(np.log(r))[:,None] , axis=0),np.gradient( np.log(mu))[None,:] ,axis=1)  + 3.
        print('# NEW FEATURE NEED TO BE INTEGRATED TO ALL FUNCTIONS')
    else:
        dlogr = np.diff(np.log(r))
        dlognr = np.diff(np.log(nr))
        d2 = dlognr/dlogr + 3.
        d2_res = np.concatenate(( np.array([d2[0]]),d2 ))

    return(d2_res)

###### New Observables ######

def nr_str(r,dd,rr,w_smallscale_corr=None):
    nr_est_gal = np.cumsum(dd)/np.cumsum(rr)
    if w_smallscale_corr!=None:
        nr_est_gal[w_smallscale_corr]=0.
    return nr_est_gal

def nr_jc(r,dd,rr,dr,who_xi='ls',w_smallscale_corr=None,corr_lss=False):
    nr_est_gal = r*0.
    xi_ls = xi(dd,rr,dr,who_xi=who_xi,corr_lss=corr_lss) 
    if w_smallscale_corr!=None:
        xi_ls[w_smallscale_corr]=0.

    xi_ls_r2 = xi_ls * r * r
    Integration = integrate.cumtrapz(xi_ls_r2,x=r,initial=xi_ls_r2[0]) #  initial=xi_ls_r2[0]
    nr_est_gal = 1 + (3./ (r**3) )*Integration

    return nr_est_gal

def nr_p(r,dd,rr,dr,who_xi='ls',w_smallscale_corr=None,corr_lss=False):
    ''' not good '''
    xi_gal = xi(dd,rr,dr,who_xi=who_xi,corr_lss=corr_lss)
    nr_est_gal = 1 + (3./2.)*xi_gal
    return nr_est_gal

def nr_jm(r,dd,rr,dr,w_smallscale_corr=None):

    xi_ls_arithm = dd-2.*dr+rr
    if np.shape(np.shape(dd))==(2,):
        N_arith = np.cumsum(np.cumsum(xi_ls_arithm,axis=0),axis=1)
        N_paran = np.cumsum(np.cumsum(rr,axis=0),axis=1)
    else:
        N_arith = np.cumsum(xi_ls_arithm)
        N_paran = np.cumsum(rr)         

    ratio = N_arith/N_paran
    if w_smallscale_corr!=None:
        ratio[w_smallscale_corr]=0.

    nr_est_gal = 1. + ratio
    return nr_est_gal

def nr_gal(r,dd,rr,dr,who_Nest='JC',who_xi='ls',correct_smallscales=False,corr_lss=False):
    if correct_smallscales: w_smallscale_corr = np.where(r<5.0) # default r<5.0
    else: w_smallscale_corr=None

    if   who_Nest=='JC': nr_est_gal = nr_jc(r,dd,rr,dr,who_xi=who_xi,w_smallscale_corr=w_smallscale_corr,corr_lss=corr_lss)
    elif who_Nest=='JM': nr_est_gal = nr_jm(r,dd,rr,dr,w_smallscale_corr=w_smallscale_corr)
    elif who_Nest=='str':nr_est_gal = nr_str(r,dd,rr,w_smallscale_corr=w_smallscale_corr)
    elif who_Nest=='p':  nr_est_gal = nr_p(r,dd,rr,dr,who_xi=who_xi,w_smallscale_corr=w_smallscale_corr,corr_lss=corr_lss)

    return nr_est_gal

############### End Observables Xi(r) N(r) and D2(r) ##########################

def rhomo_xi(r,xi):
    """New feature!"""
    w=np.where(r >= 10)
    f=interpolate.interp1d(xi[w],r[w],bounds_error=False)
    return(f(0.01))    

def rhomo_nr(r,nrvec):
    w=np.where(r >= 10)
    ther=r[w]
    thenrvec=nrvec[w]
    f=interpolate.interp1d(thenrvec[::-1],ther[::-1],bounds_error=False)
    return(f(1.01))

def rhomo_d2(r,d2):
    w=np.where(r >= 10)
    f=interpolate.interp1d(d2[w],r[w],bounds_error=False)
    return(f(2.97))

def homogeneity_many_pairs(files,   
    cosmoFID={},cosmoGAU={},biasFID=None,sigpFID=None,zmid=0.5,kaiserFID=False,dampingFID=False,galaxyFID=False,biasGAU=2.0,sigpGAU=350.,kaiserGAU=False,dampingGAU=False,galaxyGAU=False,change_gauge=True,who_xi='ls',who_Nest='JC'):
    r,dd,rr,dr,ng,nr       = read_pairs(files[0])

    nsim=np.size(files)
    nbins=np.size(r)

    all_xi  =np.zeros((nsim,nbins))
    all_nr  =np.zeros((nsim,nbins))
    all_d2  =np.zeros((nsim,nbins)) # all_d2=np.zeros((nsim,nbins-1))
    all_rhxi=np.zeros(nsim)
    all_rhnr=np.zeros(nsim)
    all_rhd2=np.zeros(nsim)

    i=0
    for files_i in files:
        r,dd,rr,dr,ng,nr=read_pairs(files_i)
        #!# r = r*0.0 + 1.02*r #try D_V/D_V 
        thexi         = xi(dd,rr,dr,who_xi=who_xi)
        
        all_xi[i,:]=thexi                #/bias[i]**2 #!# for test covariance matrix
        all_rhxi[i]=rhomo_xi(r,thexi)

        thenr=nr_gal(r,dd,rr,dr,who_Nest=who_Nest,who_xi=who_xi)

        all_nr[i,:]=thenr
        i=i+1

    if change_gauge:
        r_gal2MMgauge,all_nr_gal2MMgauge,alphas_gal2gauge = obsToMMgauge(r,all_nr,cosmoFID=cosmoFID,cosmoGAU=cosmoGAU,biasFID=biasFID,sigpFID=sigpFID,zmid=zmid,kaiserFID=kaiserFID,dampingFID=dampingFID,galaxyFID=galaxyFID,who_give='nr',biasGAU=biasGAU,sigpGAU=sigpGAU,kaiserGAU=kaiserGAU,dampingGAU=dampingGAU,galaxyGAU=galaxyGAU) # since kaiserGAU=dampingGAU=False then GAU
    else:
        all_nr_gal2MMgauge = all_nr

    for i in range(len(files)):
        
        all_nr[i,:]=all_nr_gal2MMgauge[i,:]
        all_rhnr[i]=rhomo_nr(r,all_nr[i,:])
        thed2 = d2_of_N(r,all_nr[i,:])

        all_d2[i,:]=thed2
        all_rhd2[i]=rhomo_d2(r,thed2)  #        all_rhd2[i]=rhomo_d2(r[1:],thed2)

    
    mean_xi,sig_xi,covmat_xi,cormat_xi=average_realisations(all_xi)
    mean_nr,sig_nr,covmat_nr,cormat_nr=average_realisations(all_nr)
    mean_d2,sig_d2,covmat_d2,cormat_d2=average_realisations(all_d2)

    #!# stop #try D_V/D_V 
    return(r,mean_xi,sig_xi,
             mean_nr,sig_nr,
             mean_d2,sig_d2,
             covmat_xi,covmat_nr,covmat_d2,
             cormat_xi,cormat_nr,cormat_d2,
             all_rhxi,all_rhnr,all_rhd2,
             all_xi,all_nr,all_d2)


##### just to learn how to see the double loop
def homogeneity_many_pairs_combine(filesN,filesS,bias,who_xi='ls',who_Nest='JC'):
    '''need update '''
    print 'need update'
    r,dd,rr,dr,ng,nr=read_pairs(filesN[0],normalize=True)
    nsim=np.size(filesN)
    nbins=np.size(r)
    
    all_xi=np.zeros((nsim,nbins))
    all_nr=np.zeros((nsim,nbins))
    all_d2=np.zeros((nsim,nbins))
    all_rhxi=np.zeros(nsim)
    all_rhnr=np.zeros(nsim)
    all_rhd2=np.zeros(nsim)

    i=0
    for fileN,fileS in zip(filesN,filesS):

        r,dd,rr,dr,ng,nr=read_pairs_combine_regions(fileN,fileS,normalize=True)

        thexi      =xi(dd,rr,dr,who_xi=who_xi)
        if type(bias) == np.ndarray:
            thed2,thenr=d2_nr_compute(r,dd,rr,dr,bias[i],who_Nest=who_Nest)
        else:
            thed2,thenr=d2_nr_compute(r,dd,rr,dr,bias,who_Nest=who_Nest)

        all_xi[i,:]=thexi
        all_rhxi[i]=rhomo_nr(r,thexi)

        all_nr[i,:]=thenr
        all_rhnr[i]=rhomo_nr(r,thenr)
        
        all_d2[i,:]=thed2
        all_rhd2[i]=rhomo_d2(r,thed2)
        i=i+1

    mean_xi,sig_xi,covmat_xi,cormat_xi=average_realisations(all_xi)
    mean_nr,sig_nr,covmat_nr,cormat_nr=average_realisations(all_nr)
    mean_d2,sig_d2,covmat_d2,cormat_d2=average_realisations(all_d2)

    return(r,mean_xi,sig_xi,
             mean_nr,sig_nr,
             mean_d2,sig_d2,
             covmat_xi,covmat_nr,covmat_d2,
             cormat_xi,cormat_nr,cormat_d2,
             all_rhxi,all_rhnr,all_rhd2,
             all_xi,all_nr,all_d2)
##### END just to learn how to see the double loop ####


def average_realisations(datasim):
    dims=np.shape(datasim)
    nsim=dims[0]
    nbins=dims[1]
    meansim=np.zeros(nbins)
    sigsim=np.zeros(nbins)
    for i in np.arange(nbins):
        meansim[i]=np.mean(datasim[:,i])
        sigsim[i]=np.std(datasim[:,i])
    
    covmat=np.zeros((nbins,nbins))
    for i in np.arange(nbins):
        for j in np.arange(nbins):
            covmat[i,j]=(1./(nsim))*np.sum((datasim[:,i]-meansim[i])*(datasim[:,j]-meansim[j]))

    cormat=np.zeros((nbins,nbins))
    for i in np.arange(nbins):
        for j in np.arange(nbins):
            cormat[i,j]=covmat[i,j]/np.sqrt(covmat[i,i]*covmat[j,j])

    return(meansim,sigsim,covmat,cormat)


###### CHANGING TO MM_GAUGE ############

def obsToMMgauge(r,obs,cosmoFID={},cosmoGAU={},zmid=0.5,biasFID=None,sigpFID=None,kaiserFID=False,dampingFID=False,galaxyFID=False,who_give='nr',biasGAU=2.0,sigpGAU=350.,kaiserGAU=False,dampingGAU=False,galaxyGAU=False,vShape=None):
    
    if   who_give=='xi': decalage=0.0 
    elif who_give=='nr': decalage=1.0
    elif who_give=='d2': decalage=3.0
    
    pkTheoryGAU = pk_Class.theory(cosmoGAU,zmid,halofit=False)
    obs_cosmo_GAU = getattr( pkTheoryGAU , who_give )(r,bias=biasGAU,sigp=sigpGAU,kaiser=kaiserGAU,damping=dampingGAU,galaxy=galaxyGAU)

    pkTheoryFID = pk_Class.theory(cosmoFID,zmid,halofit=False) #!#
    # this 'if-state' is for broadcasting arrays when using mocks
    if type(biasFID) == np.ndarray:
        obs_cosmo_FID_list = []
        for i in xrange(len(biasFID)):
            obs_cosmo_FID_list.append( getattr( pkTheoryFID , who_give )(r,bias=biasFID[i],sigp=sigpFID[i],kaiser=kaiserFID,damping=dampingFID,galaxy=galaxyFID) ) 
        obs_cosmo_FID = np.array(obs_cosmo_FID_list)
        alpha =  ( obs_cosmo_GAU[None,:] - decalage ) / ( obs_cosmo_FID - decalage ) # broadcasting
        obs_new = alpha * (obs-decalage) + decalage
        #figure(3)
        #for j in range(1000): plot(r,r*0+obs_new[j],'g')
    else:
        obs_cosmo_FID = getattr( pkTheoryFID , who_give )(r,bias=biasFID,sigp=sigpFID,kaiser=kaiserFID,damping=dampingFID,galaxy=galaxyFID)    
        alpha   =  (obs_cosmo_GAU - decalage) / (obs_cosmo_FID - decalage)
        obs_new = alpha * (obs - decalage) + decalage
        #figure(3),plot(r,r*0+obs_new,'r',lw=2)
        #print 'DATA_COMPL'

    return r,obs_new,alpha

###### END: CHANGING TO MM_GAUGE #######

###### Fitting d2 ###################################################

# polynomial fitting with inverse covariance matrix
def lnprobcov(thepars, xvalues, yvalues, invcovmat):
    pol=np.poly1d(thepars)
    delta=yvalues-pol(xvalues)
    chi2=np.dot(np.dot(np.transpose(delta),invcovmat),delta)
    return(-chi2)

# polynomial model
def polymodel(x,*params):
    thep=np.poly1d(params)
    return(thep(x))

def give_fitVariables(who_give='d2'):
    if who_give=='xi':
        homogenValue,ylabel_val,legend_loc,ylims,threshold=0.,'$\\xi(r)$',1,[-.01,0.05],0.01 #[-0.01,0.05],   
    elif who_give=='nr':
        homogenValue,ylabel_val,legend_loc,ylims,threshold=1.,'$\mathcal{N}(r)$',1,[0.998,1.04],1.01#[0.999,1.01],1.001 
        # [0.9,1.8]str,
        # [0.998,1.04],1.01
        # [0.999,1.01],1.001 
    elif who_give=='d2':
        homogenValue,ylabel_val,legend_loc,ylims,threshold=3.,'$\mathcal{D}_2(r)$',4,[2.92,3.01],2.97#[2.990,3.003],2.997
        # 3.,'$\mathcal{D}_2(r)$',4,[2.92,3.01],2.97
        # 3.,'$\mathcal{D}_2(r)$',4,[2.99,3.002],2.997 
        # 3.,'$\mathcal{D}_2(r)$',3,[2.96,3.002],2.97
        # 3.,'$\mathcal{D}_2(r)$',4,[2.88,3.01],2.97
        #[2.30,3.1]str,JC[2.90,3.01],[2.990,3.003],2.997,4                           [2.88,3.003]
    return homogenValue,ylabel_val,legend_loc,ylims,threshold

def get_rh_mcmc(x,y,cov,j_mock,zmid,poldeg=5,xstart=50,xstop=500,nburn=1000,nbmc=1000,nthreads=0,who_give='d2',doplot=True,diagonly=False):

    homogenValue,ylabel_val,legend_loc,ylims,thresh = give_fitVariables(who_give=who_give)

    # get desired sub array
    w=np.where((x >= xstart) & (x <= xstop))
    thex=x[w]
    they=y[w]
    yerr = np.sqrt(np.diag(cov))
    theyerr=np.sqrt(cov[w[0],w[0]])
    thecov=(cov[w[0],:])[:,w[0]]
    theinvcov=np.array(np.matrix(thecov).I)
    if diagonly:
        print('Using only diagonal part of the covariance matrix')
        theinvcov=zeros((np.size(w),np.size(w)))
        theinvcov[np.arange(np.size(w)),np.arange(np.size(w))]=1./theyerr**2
                
    # Simple polynomial fitting (no errors)
    polparsfit=np.polyfit(thex,they,poldeg)
    # 
    polfit2,parscov=opt.curve_fit(polymodel,thex,they,p0=polparsfit,sigma=theyerr)
    err_polfit2=sqrt(diagonal(parscov))

    nok=0
    while nok <= nbmc/2:
        ######################### MCMC using emcee #############################################
        nok=0
        ndim=poldeg+1
        nwalkers=ndim*2
        print('\nStart emcee with '+np.str(ndim)+' dimensions and '+np.str(nwalkers)+' walkers')
        # initial guess
        p0=emcee.utils.sample_ball(polfit,err_polfit2*3,nwalkers)
        # Initialize emcee
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobcov, args=[thex,they,theinvcov],threads=nthreads)
        # Burn out
        print('   - Burn-out with:')
        pos=p0
        okburn=0
        niterburn=0
        while okburn == 0:
            pos, prob, state = sampler.run_mcmc(pos, nburn)
            niterburn=niterburn+nburn
            chains=sampler.chain
            sz=chains[0,:,0].size
            largesig=np.zeros([nwalkers,ndim])
            smallsig=np.zeros([nwalkers,ndim])
            for j in arange(ndim):
                for i in arange(nwalkers):
                    largesig[i,j]=np.std(chains[i,sz-nburn:sz-101,j])
                    smallsig[i,j]=np.std(chains[i,sz-100:sz-1,j])
            
            ratio=largesig/smallsig
            bestratio=zeros(ndim)
            for i in arange(ndim):
                bestratio[i]=ratio[:,i].min()
                    
            worsebestratio=bestratio.max()
            wbest=np.where(bestratio == worsebestratio)
            print('     niter='+np.str(niterburn)+' : Worse ratio for best walker :'+np.str(worsebestratio))
            if worsebestratio < 2:
                okburn=1
                print('     OK burn-out done')
        
        sampler.reset()
        # now run MCMC
        print('   - MCMC with '+np.str(nbmc)+' iterations')
        pos_dif,prob_dif,state_dif=sampler.run_mcmc(pos, nbmc)
        
        # find chain for best walker
        chains=sampler.chain
        fractions=sampler.acceptance_fraction
        #########################################################################################
        frac_threshold=0.4
        print('     Best fraction: '+np.str(max(fractions)))
        wfrac=where(fractions >= frac_threshold)
        print('     '+np.str(np.size(wfrac))+' walkers are above f='+np.str(frac_threshold))
        if max(fractions) > frac_threshold:
            best=np.where(fractions == max(fractions))
            #bestwalker=best[0]
            #thechain=chains[bestwalker[0],:,:]
            thechain=chains[wfrac[0],:,:]
            sp=np.shape(thechain)
            thechain=np.reshape(thechain,[sp[0]*sp[1],sp[2]])
            # find roots
            nelts=sp[0]*sp[1]
            vals=np.zeros(nelts)
            for i in np.arange(nelts):
                roots=(np.poly1d((thechain[i,:]).flatten())-thresh).r
                w0=np.where((roots > np.min(x)) & (roots < np.max(x)) & (np.imag(roots)==0))
                if np.size(w0)==1:
                    vals[i]=np.min((np.real(roots[w0])).flatten())

            wok=where(vals != 0)
            nok=np.size(wok)
            print '(j_mock,zmid) = '+str(j_mock)+','+str(zmid)
        if nok < nbmc/2:
            print('       -> chain was not good (nok='+np.str(nok)+')... retrying...')
            print '(j_mock,zmid) = '+str(j_mock)+','+str(zmid)

    meanrh=np.mean(vals[wok])
    sigrh=np.std(vals[wok])

    bla=zeros((thex.size,nelts))
    for i in arange(nelts):
        aa=np.poly1d(thechain[i,:].flatten())
        bla[:,i]=aa(thex)

    avpol=zeros(thex.size)
    sigpol=zeros(thex.size)
    for i in arange(thex.size):
        avpol[i]=np.mean(bla[i,:])
        sigpol[i]=np.std(bla[i,:])

    ### My view of NDF ###
    NDF = thex.size-sp[2]

    ### My view of chi2 ###
    thepars = np.mean(thechain,axis=0)
    #thepars1= thechain[0]
    xvalues = thex
    yvalues = they
    Invcov  = theinvcov
    chi2    = -lnprobcov(thepars, xvalues, yvalues, Invcov)
    
    #show a plot if needed
    if(doplot):
        clf()
        subplot(2,1,1)
        #a = axes([.60, .2, .25, .25])
        hist(vals[wok],100,color='blue')
        plt.suptitle('Homogegeity scale [$h^{-1}.\mathrm{Mpc}$]')
        
        subplot(2,1,2)
        xlim(30,190)
        ylim(ylims) #ylim(2.88,3.001)
        #xlim(min(x)*30,max(x)*1.05)
        #ylim(min(they)-(3-min(they))*0.6,3+(3-min(they))*0.01)
        
        #plot([xstart,xstart],[-10,10],'--',color='black')
        #plot([xstop,xstop],[-10,10],'--',color='black')
        plot(x,x*0+homogenValue,'--',color='black')
        plot(x,x*0+thresh,'--',color='red')
        plot(thex,avpol,color='b',label='BEST fit poly (degree='+np.str(poldeg)+')')
        plot(thex,avpol+sigpol,color='b',ls=':')
        plot(thex,avpol-sigpol,color='b',ls=':')

        errorbar(x,y,yerr=yerr,fmt='ko',label='Data '+'$\chi^2/ndf = $'+str(np.around(chi2,decimals=2))+'/'+str(NDF) )
        errorbar(meanrh,thresh,xerr=sigrh,fmt='ro',label='$R_H$ = '+str('%.1f'%meanrh)+' $\pm$ '+str('%.1f'%sigrh)+' $h^{-1}\mathrm{Mpc}$')
        xlabel('r [$h^{-1}\mathrm{Mpc}$]')
        ylabel(ylabel_val)
        legend(loc=legend_loc)

    del(sampler)
    # return R_H
    returnchains=thechain[wok[0],:]
    print('Fit OK : R_H = '+str('%.1f'%meanrh)+' \pm '+str('%.1f'%sigrh)+' h^{-1}.\mathrm{Mpc}')
    
    return(meanrh,sigrh,vals[wok],returnchains,chi2,NDF)

def get_rh_spline(x,y,cov,nbspl=12,nmock_prec=None,xstart=30,xstop=1300,doplot=True,diagonly=False,logspace=True,who_give='d2',ZOOM='False',ZOOM1='False',cholesky=False,doNrTrick=True,
    QPM2PL_a=np.nan,allObsN=None,nmock=None,rth=None,obs_theory=None,rh_obs_theory=None,znames=None,
    color='b'):

    homogenValue,ylabel_val,legend_loc,ylims,threshold = give_fitVariables(who_give=who_give)

    # get desired sub array
    w=np.where((x >= xstart) & (x <= xstop))
    yerr=np.sqrt(np.diag(cov))
    thex=x[w]
    they=y[w]
    theyerr=np.sqrt(cov[w[0],w[0]])
    thecov=(cov[w[0],:])[:,w[0]]
    theinvcov=np.array(np.matrix(thecov).I)
    #theinvcov = numMath.mySVD(thecov)
    #theinvcov = linalg.pinv2(thecov)
    
    if diagonly:
        print('Using only diagonal part of the covariance matrix')
        theinvcov=zeros((np.size(w),np.size(w)))
        theinvcov[np.arange(np.size(w)),np.arange(np.size(w))]=1./theyerr**2
    
    # Fit with splines
    #print(SplineFitting.__init__)
    
    if (who_give=='nr')&(doNrTrick==True):
        thepow = 2.2
        theynew   = (they-1.)*thex**thepow + 1.
        thecovnew = np.transpose(thex**thepow) * np.transpose( thecov*(thex**thepow) )
        spl=SplineFitting.MySplineFitting(thex,theynew,thecovnew,nbspl,nmock_prec=nmock_prec,logspace=logspace)    
    else:
        spl=SplineFitting.MySplineFitting(thex,they,thecov,nbspl,nmock_prec=nmock_prec,logspace=logspace,cholesky=cholesky)
    
    # rh
    newx=linspace(thex.min(),thex.max(),1000)
    
    if (who_give=='nr')&(doNrTrick==True):
        y_spl = (spl(newx)-1.)/newx**thepow + 1.
    else:   
        y_spl =  spl(newx)

    ff=interpolate.interp1d(y_spl,newx)
    rh=ff(threshold)

    # Error Propagation on rh, wrong for (nr - 1)*r^p + 1 
    thepartial=np.zeros(spl.nbspl)
    for i in arange(spl.nbspl):
        pval=linspace(spl.alpha[i]-0.01*spl.dalpha[i],spl.alpha[i]+0.01*spl.dalpha[i],2)
        yyy=zeros(np.size(pval))
        for j in arange(np.size(pval)):
            thepars=np.copy(spl.alpha)
            thepars[i]=pval[j]
            yyy[j]=spl.with_alpha(rh,thepars)

        thepartial[i]=np.diff(yyy)/np.diff(pval)

    err_on_funct=np.sqrt(np.dot(np.dot(thepartial,spl.covout),thepartial))
    newx=np.linspace(thex.min(),thex.max(),1000)
    deriv_spl=interpolate.interp1d(newx[1:1000],np.diff(spl(newx))/np.diff(newx))
    drh=abs(err_on_funct/deriv_spl(rh))

    #show a plot if needed
    if(doplot):
        fig = figure(1)
        plotting= fig.add_subplot(111)
        plotting.tick_params('x', length=10, width=1, which='major')
        plotting.tick_params('x', length=5, width=1, which='minor')
        plt.clf()
        #figure(figsize=(15, 10))
        plt.xscale('log')
        plt.xlim(min(thex)*0.9,max(x)*1.05) #min(x+9) # min(thex)*0.9
        plt.ylim(ylims) #ylim(2.85,3.01)
        plt.plot(x,x*0+homogenValue,'--',color='black')
        plt.plot(x,x*0+threshold,'--',color='red')
        if ZOOM1: pass
        else: plot(newx,y_spl,'b-',lw=2,)
        plt.errorbar(x,y,yerr=yerr,fmt='ko',label='Best Fit Spline ('+np.str(nbspl)+' nodes): $\chi^2/\mathrm{ndf}=$'+str('%.3f'%spl.chi2)+'/'+np.str(np.size(thex)-nbspl))  #label='Data')
        #errorbar(thex,they,yerr=theyerr,fmt='yo')
        if ZOOM1: pass
        else: errorbar(rh,threshold,xerr=drh,fmt='ro',label='$R_H$ = '+str('%.1f'%rh)+' $\pm$ '+str('%.1f'%drh)+' $h^{-1}\mathrm{Mpc}$')
        plt.xlabel('r [$h^{-1}\mathrm{Mpc}$]', fontsize=15)
        plt.ylabel(ylabel_val, fontsize=20)#ylabel('$d_2(r)$')
        #suptitle('AIC = '+ str(2.*(np.size(thex)-nbspl) + spl.chi2))
        
        if np.isfinite(QPM2PL_a):
            for j in range(nmock): plt.plot(x*QPM2PL_a,allObsN[j],'g',alpha=0.01)
            plt.plot(rth,obs_theory,'r-',label='$\mathcal{D}^{th}_2(r)$ $\mathcal{R}^{th}_{H} = $ %.1f $h^{-1}\mathrm{Mpc}$'%rh_obs_theory)
            plt.errorbar(rh,threshold,xerr=drh,fmt='ro')
            plt.title(doNrTrick+' - '+znames, fontsize=20)
        plt.legend(loc=legend_loc, fontsize=15,frameon=False,numpoints=1)

        # ZOOM in large scales
        if ZOOM1=='True':
            if   who_give=='d2': axes([.48,0.2,.4,.4])
            elif who_give=='nr': axes([.48,0.2,.4,.4])
            xscale('log')
            xlim(90,1300.) #min(x+9) # min(thex)*0.9
            if   who_give=='d2': ylim(2.990,3.005)
            elif who_give=='nr': ylim(0.998,1.005) #ylim(2.85,3.01)
            plot(x,x*0+homogenValue,'--',color='black')
            plot(x,x*0+threshold,'--',color='red')
            plot(newx,y_spl,color='b',lw=2,label='Best Fit Spline ('+np.str(nbspl)+' nodes): $\chi^2/ndf=$'+str('%.3f'%spl.chi2)+'/'+np.str(np.size(thex)-nbspl))
            errorbar(x,y,yerr=yerr,fmt='ko',label='Data')
            #errorbar(thex,they,yerr=theyerr,fmt='yo')
            errorbar(rh,threshold,xerr=drh,fmt='ro',label='$R_H$ = '+str('%.1f'%rh)+' $\pm$ '+str('%.1f'%drh)+' $h^{-1}\mathrm{Mpc}$')
            #xlabel('r [$h^{-1}\mathrm{Mpc}$]', fontsize=15)
            #ylabel(ylabel_val, fontsize=20)#ylabel('$d_2(r)$')
        if ZOOM=='True':
            if   who_give=='d2': axes([.48,0.42,.4,.25])
            elif who_give=='nr': axes([.48,0.23,.4,.25])
            xscale('linear')
            xlim(50,80.) #xlim(50,80.) matter ,xlim(90,200.) galaxies   #min(x+9) # min(thex)*0.9
            if   who_give=='d2': ylim(2.96,2.98)
            elif who_give=='nr': ylim(1.006,1.016) #ylim(2.85,3.01)
            plot(x,x*0+homogenValue,'--',color='black')
            plot(x,x*0+threshold,'--',color='red')
            plot(newx,y_spl,color='b',lw=2,label='Best Fit Spline ('+np.str(nbspl)+' nodes): $\chi^2/ndf=$'+str('%.3f'%spl.chi2)+'/'+np.str(np.size(thex)-nbspl))
            errorbar(x,y,yerr=yerr,fmt='ko',label='Data')
            #errorbar(thex,they,yerr=theyerr,fmt='yo')
            errorbar(rh,threshold,xerr=drh,fmt='ro',label='$R_H$ = '+str('%.1f'%rh)+' $\pm$ '+str('%.1f'%drh)+' $h^{-1}\mathrm{Mpc}$')
            #xlabel('r [$h^{-1}\mathrm{Mpc}$]', fontsize=15)
            #ylabel(ylabel_val, fontsize=20)#ylabel('$d_2(r)$')
        draw()
        show()

    print('Fit OK : R_H = '+str('%.1f'%rh)+' \pm '+str('%.1f'%drh)+' h^{-1}.\mathrm{Mpc} \chi2='+str('%.1f'%spl.chi2)+'/'+str('%.1f'%(np.size(thex)-nbspl)))
    return(rh,drh,spl) 

def get_a_modelTheory(x,y,cov,QPM2PL_a=1.0,functmodel=polymodel,xstart=40.,xstop=1300.,nmock_prec=None,method='minuit',doplot=False,doplotPULL=False,randVal=4e-7):
    wok = np.where((x>xstart)&(x<xstop))
    newx   = x[wok]*QPM2PL_a
    newy   = y[wok]
    newcov = (cov[wok[0],:])[:,wok[0]]+np.diag(np.random.rand(len(newy))*randVal)
    #newcov = (cov[wok[0],:])[:,wok[0]]+np.diag(np.zeros(len(newy))+randVal)

    #figure(2),clf()
    #plot( newx,np.diag( (cov[wok[0],:])[:,wok[0]]) ),plot(newx,np.diag(newcov))
    fitTheory = fitting.dothefit(newx,newy,newcov,[0.998],functname=functmodel,method=method,parbounds=[(0.80,1.2)],nmock_prec=nmock_prec)

    if doplotPULL:
        pull = numMath.pullCov(newy,functmodel(newx,fitTheory[1]),newcov)

    if doplot: 
        decalage=0. # to show result better on large scales r>200
        figure(1),plt.clf() 
        if doplotPULL: plt.subplot(2,1,1)    
        plt.xscale('log'),plt.xlabel('r [$h^{-1}\mathrm{Mpc}$]', fontsize=15)
        
        plt.xlim(xstart*0.9,xstop*1.1),plt.ylim(min(newy)*0.999,max(newy)*1.001),plt.plot(x,x*0+3,'k--')
        #plt.xlim(40.,1300.),plt.ylim(0.99,1.02),plt.plot(x,x*0+1,'k--')
        if decalage!=0.: plt.xlim(xstart*0.9,xstop*1.1),plt.ylim(-0.004,0.002),plt.plot(x,x*0,'k--') 
        
        plt.ylabel('$\mathcal{D}_2(r)$')
        if decalage!=0.:         plt.ylabel('$\mathcal{D}_2(r)-3$')

        plt.errorbar(x,y-decalage,yerr=np.sqrt( np.diag(cov) ), fmt='bo' )
        label=' $\chi^2$ = %0.1f / %0.0f'%(fitTheory[4],fitTheory[5])+' \n $a$= %0.3f $\pm$ %0.3f'%(fitTheory[1],fitTheory[2])
        plt.plot(x,functmodel(x,fitTheory[1])-decalage,'r.-',label=label)
        plt.legend(loc=4,numpoints=1,frameon=False)
        plt.draw()
        if doplotPULL: 
            plt.subplot(2,1,2)
            hist(pull,label='%0.3f$\pm$%0.3f'%(np.mean(pull),np.std(pull)))
            plt.legend(loc=1,numpoints=1,frameon=False)
        plt.draw()

    return fitTheory[4],fitTheory[5],fitTheory[1],fitTheory[2]

def get_chi2_LCDM(x,y,covarin,QPM2PL_a=1.0,functmodel=polymodel,xstart=40.,xstop=1300.,nmock_prec=None,doplot=False,doplotPULL=False):
    ''' brute force chi2 calc '''

    wok = np.where((x>xstart)&(x<xstop))
    newx   = x[wok]*QPM2PL_a
    newy   = y[wok]
    newcov = (cov[wok[0],:])[:,wok[0]]+np.diag(np.random.rand(len(newy))*randVal)

    if nmock_prec!=None: 
        precision_fctor_for_cov = (nmock_prec-1.)/(nmock_prec-len(newx)-2.)
        print precision_factor_for_cov
        newcov=newcov * precision_factor_for_cov

    aas = np.linspace(0.80,1.0,1e2)

    chi2_a = np.zeros_like(aas)

    chi2_instance = fitting.MyChi2(newx,newy,newcov,functmodel)
    for i in xrange(len(aas)): chi2_a[i] = chi2_instance(aas[i])


    chi2min     = min(chi2_a)
    wok_chi2min = np.where(chi2_a==chi2min)
    par_give    = aas[wok_chi2min]
    par_left = max(aas[np.where((chi2_a<chi2min+1)&(aas<par_give))])
    par_right= max(aas[np.where((chi2_a<chi2min+1)&(aas>par_give))])

    dpar_give = abs(par_right-par_left)

    if doplot:
        x = np.linspace(min(aas),max(aas),5)
        plt.plot(x,x*0.0 + (len(newx)-1),'k--')
        label = '$a=$%0.03f$\pm$%0.3f \n $\chi^2=$%0.1f'%(par_give,dpar_give,chi2min)
        plt.plot(aas,chi2_a,'-',label=label)
        plt.legend(loc=1,)
        plt.draw()

    return chi2min,len(newx-1),par_give,dpar_give

def get_Transition0(x,y,covarin,QPM2PL_a=1.0,functmodel=polymodel,xstart=200.,xstop=1300.,nmock_prec=None,method='minuit',doplot=False,doplotPULL=False,randVal=2e-7,who_give='d2'):

    if   who_give=='d2': decalage = 3.
    elif who_give=='nr': decalage = 1.
    elif who_give=='xi': decalage = 0.

    wok = np.where((x>xstart)&(x<xstop))
    newx   = x[wok]*QPM2PL_a
    newy   = y[wok]
    newcov = (covarin[wok[0],:])[:,wok[0]]

    delta = newy-decalage
    chi2  = np.dot( delta.T ,np.dot(np.linalg.inv(newcov),delta) )

    if doplotPULL:
        pull = numMath.pullCov(newy,np.mean(newy),newcov)

    if doplot:
        plt.clf()
        if doplotPULL: plt.subplot(2,1,1)    
        plt.xscale('log'),plt.xlabel('r [$h^{-1}\mathrm{Mpc}$]', fontsize=15)
        plt.xlim([xstart*0.99,xstop*1.01]),plt.ylim([-0.005,0.005])
        plt.errorbar(newx,delta,yerr=np.sqrt(np.diag(newcov)),fmt='bo',label='$\chi^2=$%0.3f$/$%0.0f'%(chi2,len(newx)))
        plt.plot(newx,newx*0.0,'k--')
        plt.legend(loc=4,numpoints=1,frameon=False)
        plt.ylabel('$\mathcal{D}_2(r)$')
        if decalage!=0.:  plt.ylabel('$\mathcal{D}_2(r)-3$')
        if doplotPULL: 
            plt.subplot(2,1,2)
            hist(pull,label='%0.3f$\pm$%0.3f'%(np.mean(pull),np.std(pull)))
            plt.legend(loc=1,numpoints=1,frameon=False)
    # 1,2,3 is to match the previous function output: get_chi2_LCDM
    return chi2,1,2,3

def read_data(datafile,who_xi='ls',who_Nest='JC',combine=False,corrSS = False):
    # Read data
    if combine is False:
        r,dd,rr,dr,ngal,nrand=read_pairs(datafile)
    elif combine is True:
        r,dd,rr,dr,ngal,nrand=read_pairs_combine_regions(datafile[0],datafile[1])
    elif combine is 'cute':
        r,xi0,dd,rr,dr,ngal,nrand=read_cute(datafile)
        corrSS = True
    elif combine is 'cute_combine':
        r,xi0,dd,rr,dr,ngal,nrand=read_cute_combine_regions(datafile[0],datafile[1])
        corrSS = True
    
    twoPCF = xi(dd,rr,dr,who_xi=who_xi) # galaxy 2PCF
    # calculate n(r) and d2(r)
    nr = nr_gal(r,dd,rr,dr,who_xi=who_xi,who_Nest=who_Nest,correct_smallscales=corrSS)
    d2 = d2_of_N(r,nr) # gal or DM depending on bias
    
    return(r,d2,nr,twoPCF)

def read_mocks(mockdir, cosmoFID={},cosmoGAU={},zmid=0.5,
    change_gauge=True,biasFID=None,sigpFID=None,kaiserFID=False,dampingFID=False,galaxyFID=False,biasGAU=2.0,sigpGAU=350.,kaiserGAU=False,dampingGAU=False,galaxyGAU=False, 
    who_xi='ls',who_Nest='JC',who_give='d2',pairfiles='pairs_*.txt',combine=False,calc_xi=True,xi0Bool=True,change_gauge_xi=False):
    # Read Mocks to get covariance matrix
    
    if mockdir is None:
        covmatd2=np.zeros((np.size(rd),np.size(rd)))
        covmatd2[np.arange(np.size(rd)),np.arange(np.size(rd))]=1e-4
    else:
        if combine is False:
            mockfiles=glob.glob(mockdir+pairfiles) 
            nmock = np.shape(mockfiles)[0]
            #mockfiles = mockfiles_[:100] # 990 pour debugging
            #r,mean_xi,sig_xi,mean_nr,sig_nr,mean_d2,sig_d2,covmat_xi,covmat_nr,covmat_d2,cormat_xi,cormat_nr,cormat_d2,rhxi,rhnr,rhd2,sigrhxi,sigrhnr,sigrhd2,all_xi,all_nr,all_d2=homogeneity_many_pairs(mockfiles,bias,who_Nest=who_Nest,precision=precision)
            r,mean_xi,sig_xi,mean_nr,sig_nr,mean_d2,sig_d2,covmat_xi,covmat_nr,covmat_d2,cormat_xi,cormat_nr,cormat_d2,all_rhxi,all_rhnr,all_rhd2,all_xi,all_nr,all_d2=homogeneity_many_pairs(mockfiles,   cosmoFID=cosmoFID,cosmoGAU=cosmoGAU,biasFID=biasFID,sigpFID=sigpFID,zmid=zmid,kaiserFID=kaiserFID,dampingFID=dampingFID,galaxyFID=galaxyFID,biasGAU=biasGAU,sigpGAU=sigpGAU,kaiserGAU=kaiserGAU,dampingGAU=dampingGAU,galaxyGAU=galaxyGAU,change_gauge=change_gauge,who_xi=who_xi,who_Nest=who_Nest)

        elif combine is True:
            mockfiles0=glob.glob(mockdir[0]+pairfiles)
            mockfiles1=glob.glob(mockdir[1]+pairfiles)
            nbcommon=min([size(mockfiles0),size(mockfiles1)])
            print 'wont work need update' 
            #r,mean_xi,sig_xi,mean_nr,sig_nr,mean_d2,sig_d2,covmat_xi,covmat_nr,covmat_d2,cormat_xi,cormat_nr,cormat_d2,rhxi,rhnr,rhd2,sigrhxi,sigrhnr,sigrhd2,all_xi,all_nr,all_d2=homogeneity_many_pairs_combine(mockfiles0[:nbcommon],mockfiles1[:nbcommon],bias,who_Nest=who_Nest,precision=precision)
            r,mean_xi,sig_xi,mean_nr,sig_nr,mean_d2,sig_d2,covmat_xi,covmat_nr,covmat_d2,cormat_xi,cormat_nr,cormat_d2,all_rhxi,all_rhnr,all_rhd2,all_xi,all_nr,all_d2=homogeneity_many_pairs_combine(mockfiles0[:nbcommon],mockfiles1[:nbcommon],bias,who_Nest=who_Nest,precision=precision)
        elif combine is 'cute':
            mockfiles=glob.glob(mockdir+pairfiles) 
            nmock = np.shape(mockfiles)[0]
            r,mean_xi,sig_xi,mean_nr,sig_nr,mean_d2,sig_d2,covmat_xi,covmat_nr,covmat_d2,cormat_xi,cormat_nr,cormat_d2,all_rhxi,all_rhnr,all_rhd2,all_xi,all_nr,all_d2=read_cute_many(mockfiles, 
                cosmoFID=cosmoFID,cosmoGAU=cosmoGAU,biasFID=biasFID,sigpFID=sigpFID,zmid=zmid,kaiserFID=kaiserFID,dampingFID=dampingFID,galaxyFID=galaxyFID,biasGAU=biasGAU,sigpGAU=sigpGAU,kaiserGAU=kaiserGAU,dampingGAU=dampingGAU,galaxyGAU=galaxyGAU,change_gauge=change_gauge,
                who_xi=who_xi,who_Nest=who_Nest,calc_xi=calc_xi,change_gauge_xi=change_gauge_xi)
        elif combine is 'cute_rm':
            mockfiles=glob.glob(mockdir+pairfiles) 
            nmock = np.shape(mockfiles)[0]
            r,mean_xi,sig_xi,mean_nr,sig_nr,mean_d2,sig_d2,covmat_xi,covmat_nr,covmat_d2,cormat_xi,cormat_nr,cormat_d2,all_rhxi,all_rhnr,all_rhd2,all_xi,all_nr,all_d2=read_cute_many_rm(mockfiles, 
                cosmoFID=cosmoFID,cosmoGAU=cosmoGAU,biasFID=biasFID,sigpFID=sigpFID,zmid=zmid,kaiserFID=kaiserFID,dampingFID=dampingFID,galaxyFID=galaxyFID,biasGAU=biasGAU,sigpGAU=sigpGAU,kaiserGAU=kaiserGAU,dampingGAU=dampingGAU,galaxyGAU=galaxyGAU,change_gauge=change_gauge,
                who_xi=who_xi,who_Nest=who_Nest,calc_xi=calc_xi,xi0Bool=xi0Bool)         
        elif combine is 'cute_se':
            mockfiles=glob.glob(mockdir+pairfiles) 
            nmock = np.shape(mockfiles)[0]
            r,mean_xi,sig_xi,mean_se,sig_se,mean_ss,sig_ss,covmat_xi,covmat_se,covmat_ss,cormat_xi,cormat_se,cormat_ss,all_rhxi,all_rhse,all_rhss,all_xi,all_se,all_ss=read_cute_many_se(mockfiles, 
                cosmoFID=cosmoFID,cosmoGAU=cosmoGAU,biasFID=biasFID,sigpFID=sigpFID,zmid=zmid,kaiserFID=kaiserFID,dampingFID=dampingFID,galaxyFID=galaxyFID,biasGAU=biasGAU,sigpGAU=sigpGAU,kaiserGAU=kaiserGAU,dampingGAU=dampingGAU,galaxyGAU=galaxyGAU,change_gauge=change_gauge,
                who_xi=who_xi,who_Nest=who_Nest,calc_xi=calc_xi)           
            
            #read_cute_many(mockfiles,who_xi=who_xi,who_Nest=who_Nest,calc_xi=False) 
    if who_give=='xi':
        return(covmat_xi,mean_xi,all_rhxi,all_xi,r,nmock)
    elif who_give=='nr':
        return(covmat_nr,mean_nr,all_rhnr,all_nr,r,nmock)
    elif who_give=='d2':
        return(covmat_d2,mean_d2,all_rhd2,all_d2,r,nmock)
    elif who_give=='se':
        return(covmat_se,mean_se,all_rhse,all_se,r,nmock)
    elif who_give=='ss':
        return(covmat_ss,mean_ss,all_rhss,all_ss,r,nmock)


def read_datamocks(datafile,mockdir,who_give='d2',who_xi='ls',who_Nest='JC',bias=2.,combine=False):
    r,d2,nr,xi=read_data(datafile,who_xi=who_xi,combine=combine)
    covmatObs,mean_Obs,all_rhObs,all_obs,rrr=read_mocks(mockdir,bias,who_give=who_give,who_Nest=who_Nest,combine=combine)
    '''need update'''
    print 'need update'
    stop
    if who_give=='xi': obs=xi
    elif who_give=='nr': obs=nr
    elif who_give=='d2': obs=d2                   
    return(r,obs,covmatObs,np.mean(all_rhObs))

def getd2_datamocks(datafile,covmat_y,
    cosmoFID={},cosmoGAU={},biasFID=2.0,sigpFID=350.,change_gauge=True,zmid=0.5,kaiserFID=False,dampingFID=False,galaxyFID=False,who_Nest='JC',
    who_give='d2',combine=False,doNrTrick=True,r0=30.,rstop=1300.,nbspl=12,nmock_prec=None,doplot=False,ZOOM=False,
    ZOOM1=False,QPM2PL_a=np.nan,allObsN=None,nmock=None,rth=None,obs_theory=None,rh_obs_theory=None,znames=None
    ):
    '''NOT GOOD for datafile=mocks '''
    #read data and mocks 
    r,d2_gal,nr_gal,xi_gal=read_data(datafile,who_Nest=who_Nest,combine=combine)

    data_x = r    
    
    if change_gauge:
        radious, nr_MM ,alpha_FID2GAU = obsToMMgauge(data_x,nr_gal,cosmoFID=cosmoFID,cosmoGAU=cosmoGAU,biasFID=biasFID,sigpFID=sigpFID,zmid=zmid,kaiserFID=kaiserFID,dampingFID=dampingFID,galaxyFID=galaxyFID,who_give='nr')

        if len(np.ndarray.flatten(nr_gal))==len(xi_gal): 
            d2_MM = d2_of_N(data_x,nr_MM)
        else:
            d2_MM = np.zeros_like(nr_MM)
            for d2_i in range(len(d2_MM)):
                d2_MM[d2_i] = d2_of_N(data_x,nr_MM[d2_i])
    else:
        d2_MM,nr_MM = d2_gal,nr_gal 


    if who_give=='xi':
        data_y = xi_gal
        print 'getd2_datamocks gives xi_gal'
        stop_and_debug
    elif who_give=='nr':
        data_y = nr_MM
    elif who_give=='d2':
        data_y = d2_MM 

    if type(biasFID) == np.ndarray: # len(data_y) == len(biasFID) == len(sigpFID) 
        rh = np.zeros(np.shape(data_y)[0])
        drh= np.zeros(np.shape(data_y)[0])
        for bi in range(np.shape(data_y)[0]):
            rh[bi],drh[bi],result=get_rh_spline(data_x,data_y[bi],covmat_y,nbspl=nbspl,nmock_prec=nmock_prec,xstart=r0,xstop=rstop,who_give=who_give,doplot=doplot,ZOOM=ZOOM,doNrTrick=doNrTrick)
    else:
        rh,drh,result=get_rh_spline(data_x,data_y,covmat_y,nbspl=nbspl,nmock_prec=nmock_prec,xstart=r0,xstop=rstop,who_give=who_give,doplot=doplot,doNrTrick=doNrTrick,ZOOM1=ZOOM1,ZOOM=ZOOM, #!# Change
            QPM2PL_a=QPM2PL_a,allObsN=allObsN,nmock=nmock,rth=rth,obs_theory=obs_theory,rh_obs_theory=rh_obs_theory,znames=znames
            )
    print 'res from get_rh_spline',result
    
    return(rh,drh,result,data_x,data_y)

def getd2_datamocks_model(datafile,covmat_y,cosmo,zmid,who_give='d2',who_Nest='JC',bias=2.,combine=False,r0=30,rstop=1300,nbspl=12,doplot=False):
    #read data and mocks
    r,d2,nr,xi=read_data(datafile,bias=bias,who_Nest=who_Nest,combine=combine)

    data_x = r    
    if who_give=='xi':
        data_y = xi
    elif who_give=='nr':
        data_y = nr
    elif who_give=='d2':
        data_y = d2

    rh,drh,result=get_rh_model(data_x,data_y,covmat_y,cosmo,zmid,nbspl=nbspl,xstart=r0,xstop=rstop,who_give=who_give,doplot=doplot)
    print 'res from get_rh_model',result
    
    return(rh,drh,result,data_x,data_y)

def getd2_datamocks_mcmc(datafile,covmat_y,j_mock,zmid,who_give='d2',who_Nest='JC',bias=2.,combine=False,deg=7,r0=30,rstop=1300,nbmc=5000,doplot=False):
    #read data and mocks
    r,d2,nr,xi=read_data(datafile,bias=bias,who_Nest=who_Nest,combine=combine)

    data_x = r    
    if who_give=='xi':
        data_y = xi
    elif who_give=='nr':
        data_y = nr
    elif who_give=='d2':
        data_y = d2

    rh,drh,vals_good_rh,returnchains,chi2,NDF=get_rh_mcmc(data_x,data_y,covmat_y,j_mock,zmid,poldeg=deg,xstart=r0,xstop=rstop,nbmc=nbmc,who_give=who_give,doplot=doplot)
    
    return(rh,drh,chi2,NDF,data_x,data_y)

def give_obs_mocks(who_give='xi', who_xi='ph',who_Nest='str',mockfiles=[]):
    r,mean_xi,sig_xi,mean_nr,sig_nr,mean_d2,sig_d2,covmat_xi,covmat_nr,covmat_d2,cormat_xi,cormat_nr,cormat_d2,all_rhxi,all_rhnr,all_rhd2,all_xi,all_nr,all_d2=homogeneity_many_pairs(mockfiles,who_xi=who_xi,who_Nest=who_Nest)
    if who_give == 'xi':
        return(r,mean_xi/(bias**2),covmat_xi/(bias**2),sig_xi/(bias**2),all_xi/(bias**2),all_rhxi/(bias**2))
    elif who_give == 'xi_gal':
        return(r,mean_xi,covmat_xi,sig_xi,all_xi,all_rhxi)
    elif who_give == 'nr':
        return(r,mean_nr,covmat_nr,sig_nr,all_nr,all_rhnr) 
    elif who_give == 'd2':
        return(r,mean_d2,covmat_d2,sig_d2,all_d2,all_rhd2)

def NameZ(minz,maxz,nbins):
    """
    # returns 2 arrays which contain
    # the edges of bins of redshift
    # in double and string array 
    # used in an old version of files, possibly: DR11-DR10??
    """
    zedge = np.linspace(minz,maxz,nbins+1)
    znames = np.array(np.zeros(nbins), dtype='|S20')
    
    for i in np.arange(nbins):
        znames[i]='z_'+str(zedge[i])+'_'+str(zedge[i+1])

    return(zedge,znames)

def NameZ2(minz,maxz,nbins):
    """
    returns 2 arrays which contain
    the edges of bins of redshift
    in double and string array 
    + 3 more arrays
    USE: zedge,znames,dz,zmid,zbins = galtools.NameZ2(minz=0.43,maxz=0.7,nbins=5)
    """
    zedge = np.linspace(minz,maxz,nbins+1)
    znames = np.array(np.zeros(nbins), dtype='|S20')
    
    for i in np.arange(nbins):
        znames[i]='z_'+str(zedge[i])+'_'+str(zedge[i+1])
    
    dz=np.zeros(nbins)+(zedge[1]-zedge[0])/2
    zmid=(zedge[np.arange(nbins)]+zedge[np.arange(nbins)+1])/2

    return(zedge,znames,dz,zmid,nbins)


def corrCoeffd2(covmatd2,r,mu=None,corrfactor=1.,nbspl=9,name='',save=False,Zmin=-1.,Zmax=1.):
    #from matplotlib.colors import LogNorm

    cc = covmatd2*0.0+covmatd2*corrfactor

    """
    w_neg = np.where(covmatd2<=0.)
    cc_masked = covmatd2*0.0 + covmatd2

    cc_masked[w_neg] = covmatd2.max()*10.

    cc = abs(cc_masked)
    """
    
    '''
    cc = np.ones((np.size(r),np.size(r)))

    for i in range(np.size(r)):
        for j in range(np.size(r)):
            cc[i][j] = covmatd2[i][j]/np.sqrt(covmatd2[i][i]*covmatd2[j][j])
    '''
    #X,Y = meshgrid(rS,rS)
    

    X = r
    if mu==None: Y = r
    else: Y = mu
    Z = cc  

    #Z1=cc1
    print r.shape
    print cc.shape
    fig, ax = plt.subplots()
    if mu==None: plt.ylabel('$r_i\ [h^{-1}\ \mathrm{Mpc}]$',fontsize=30)
    else:  plt.ylabel('$\mu$ ')
    plt.xlabel('$r_j\ [h^{-1}\ \mathrm{Mpc}]$',fontsize=30)
    if mu==None: 
        plt.yscale('log')
    else: pass
    print Y,X
    plt.xscale('log')

    if mu==None: 
        if X.min()<1.:
            rminimum = 1.
        else:
            rminimum = r.min()        
    else: rminimum = r.min()         #

    if mu==None:
        plt.xlim(rminimum,r.max()) #1300) #plt.xlim(0.49,1300)
        plt.ylim(rminimum,r.max()) #1300) #plt.ylim(0.49,1300)
    else: 
        plt.xlim(10.,X.max()) # plt.xlim(X.min(),X.max()) # 
        plt.ylim(Y.min(),Y.max()) # plt.ylim(Y.min(),Y.max()) # 

    from matplotlib.colors import LogNorm
    p = ax.pcolor(X, Y, Z, cmap=cm.RdBu , vmin=Zmin,vmax=Zmax )  #Z.min(),vmax=0.1 ) if mu!=None
                                                                #norm=LogNorm(vmin=0*Z.min()+1e-5, vmax=Z.max()) ) 
                                                                 # min,max=(-3.5e-7,0.8e-5 )
                                                                 # norm=LogNorm(vmin=1e-6, vmax=1e-9)
                                                                 # vmin=Z.min(),vmax=Z.max() 
                                                                 # vmin=Zmin , vmax=Zmax 
                                                                 # norm=LogNorm(vmin=Z.min(), vmax=Z.max())
                                                                 # vmin=Z.min(),vmax=1e-2 read_cor_3D_rm.py (32,8,_linear_0_300)
    #p1 = ax.pcolor(X, Y, Z1, cmap=cm.RdBu , vmin=Z.min(),vmax=Z.max() ) 
    cb = fig.colorbar(p, ax=ax)
    cb.ax.tick_params(labelsize=25)
    
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    #plt.suptitle(name+' at $0.646<$ z $<0.7$')
    plt.suptitle(name,fontsize=25)
    if save == True:
        print 'Saving ...'
        plt.savefig('images/corr_z_0.646_0.7_'+name+'_nbspl_'+str(nbspl)+'.png',dpi=100)

    return None

def builtDir(outdir='',region=''):
    subprocess.call(["mkdir",outdir])
    subprocess.call(["mkdir",outdir+region])

def nbNames(num,flag=False):
    """
    (Default) convert 1 to  001 for flag=False
              convert 1 to 0001 for flag=True
    """
    if flag==False:
        if(num < 10):
            a1 = '00'+str(num)
            return(a1)
        elif(10<=num<100):
            a2 = '0'+str(num)
            return(a2)
        else:
            a3 = str(num)
            return(a3)
    else:
        if(num < 10):
            b1 = '000'+str(num)
            return(b1)
        elif(10<=num<100):
            b2 = '00'+str(num)
            return(b2)
        elif(100<=num<1000):
            b3 = '0'+str(num)
            return(b3)
        else:
            b4 = str(num)
            return(b4)

def factorCorr(x,pars):
    '''takes only 1st 3 params of fit'''
    fac = ( 1. + np.exp(pars[2]*x**2+pars[1]*x+pars[0]) )
    return fac

def factor_for_opening_cov(xin,yin,covmat,xstart=1.,xstop=40.,cosmoparams={},zmid=0.5,nmock_prec=None,method='minuit',doplot=False,FullModel=False,BOUNDS=False):

    pkTheory = pk_Class.theory(cosmoparams,zmid,halofit=False)

    wok= np.where((xin>xstart)&(xin<xstop))
    r_fit = xin[wok]
    cov_fit = (covmat[wok[0],:])[:,wok[0]]
    y_fit = yin[wok]

    def model_xi_x_factor(x,pars):
        bias = pars[3]
        if FullModel == True:     
            sigp = pars[4]
            y_th = factorCorr(x,pars[:-1]) * pkTheory.xi(x,bias=bias,sigp=sigp,kaiser=True,damping=True,galaxy=True) 
        elif FullModel==False: 
            y_th = factorCorr(x,pars[:-1]) * pkTheory.xi(x,bias=bias,galaxy=True)
        return y_th

    if FullModel== True: 
        guess=[1.,.1,-0.001,2.,300.]
        bounds=[ (-4.,-3.),(0.2,0.6),(-0.05,-0.01),(1.8,3.),(None,None) ]  
    else: 
        guess=[1.,.1,-0.001,2.]
        
        if BOUNDS==True:        bounds=[ (-5.,-3.),(0.1,0.7),(-0.05,-0.005),(1.5,3.) ]
        elif BOUNDS=='cute_rm' or BOUNDS=='cute': bounds=[ (-6.,-2.),(0.05,0.65),(-0.1,-0.01),(1.5,3.) ]
        elif BOUNDS==False:     bounds=[ (None,None),(None,None),(None,None),(None,None) ]

    fit = fitting.dothefit(r_fit,y_fit,cov_fit,guess,functname=model_xi_x_factor,nmock_prec=nmock_prec,method=method,parbounds=bounds)

    if doplot:
        plt.figure(3),plt.clf()
        plt.xscale('log')
        plt.xlabel('$r\ [h^{-1}Mpc]$'),plt.ylabel('$r\\xi(r)$')
        plt.errorbar(r_fit,y_fit*r_fit,yerr=np.sqrt(np.diagonal(cov_fit))*r_fit,fmt='b.')
        plt.plot(r_fit,model_xi_x_factor(r_fit,fit[1])*r_fit,'r-')
        plt.title('z = %0.3f'%zmid+', $\chi^2$ = %0.1f'%fit[4]+'/ %0.0f'%fit[5])
        plt.draw()

        plt.figure(4)#,plt.clf()
        plt.xlabel('$r\ [h^{-1}Mpc]$',size=20),plt.ylabel('$\sqrt{ 1+ \Delta^2(r)e^2}$',size=20)
        plt.plot(r_fit,sqrt(1 + (10.*(factorCorr(r_fit,fit[1][:3])-1.))**2),label='z = %0.3f '%zmid)#+str(fit[1][:3]))
        plt.legend(numpoints=1,frameon=False)
        plt.draw()

    return fit[1][:3]


def open_cov_with_model_sys(xin,yin,covmat,eff=0.0,xstart=1.,xstop=40.,cosmoparams={},zmid=0.5,nmock_prec=None,method='minuit',doplot=False,FullModel=False,flattened=False,xlim_flattened=1300.,BOUNDS=False):

    params_factor = factor_for_opening_cov(xin,yin,covmat,xstart=xstart,xstop=xstop,cosmoparams=cosmoparams,zmid=zmid,nmock_prec=nmock_prec,method=method,doplot=doplot,FullModel=FullModel,BOUNDS=BOUNDS)
    
    Delta = factorCorr(xin,params_factor) - 1.
    if flattened: Delta = numMath.flattening_func(xin,Delta,xlim=xlim_flattened,fval=0.0)

    covmat_opened = numMath.open_cov(covmat,Delta*eff)
    
    return covmat_opened,Delta

### FRACTALS ANALYSIS

def xyz2fits(fitsname,x,y,z):
    col0=afits.Column(name='x',format='E',array=x)
    col1=afits.Column(name='y',format='E',array=y)
    col2=afits.Column(name='z',format='E',array=z)
    cols=afits.ColDefs([col0,col1,col2])
    tbhdu=afits.new_table(cols) #.BinTableHDU.from_columns(cols)
    tbhdu.writeto(fitsname,clobber=True)

def zdecra2fits(fitsname,z,dec,ra):
    col0=afits.Column(name='z',format='E',array=z)
    col1=afits.Column(name='dec',format='E',array=dec)
    col2=afits.Column(name='ra',format='E',array=ra)
    cols=afits.ColDefs([col0,col1,col2])
    tbhdu=afits.new_table(cols) #.BinTableHDU.from_columns(cols)
    tbhdu.writeto(fitsname,clobber=True)

def x2y(x,y):
    ''' fractal2random position translation'''
    xmid = ( x.max() + x.min() )*0.5
    ymid = ( y.max() + y.min() )*0.5
    delta = (xmid-ymid)
    print delta , xmid, ymid
    return x-delta

def get_condition_mask(my_Data,mngMask):
    ''' my_Data is from .fits 
        mngMask is from mangle soft '''
    poly_ids = mngMask.get_polyids(my_Data['ra'] ,my_Data['dec'])
    #counter=np.sum(poly_ids != -1) #--> give you the number of objects within the mask
    my_Data_with_Mask = my_Data[poly_ids != -1]
    return my_Data_with_Mask

def write_to_file(file,Zarray,DECarray,RAarray,Warray=None):
    if Warray is None: Warray = np.ones(len(Zarray))
    outfile=open(file,'w')
    for redshift,declination,rightAscesion,weight in zip(Zarray,DECarray,RAarray,Warray):
        outfile.write("%s %s %s %s \n"%(redshift,declination,rightAscesion,weight))
    outfile.close()

### FRACTALS ANALYSIS AWESOME DAY!!

### PAIRS CUTE

def test_nr_sample(dd):
    ddint = dd*0.0
    ddint[0] = dd[0]
    for i in np.arange(dd.size-1) + 1 :
        ddint[i] = ddint[i-1]+dd[i]
    
    ddsum = np.cumsum(dd)
    
    print ddsum == ddint
    return ddint,ddsum


def read_cute(filename,normalize=True,give_rightedge=False):
    blas = np.loadtxt(filename,skiprows=1)
    if give_rightedge: addReachRedge=0.5*(1500./50.) 
    else: addReachRedge=0.0
    r    = blas[:,0] + addReachRedge
    xi0  = blas[:,1]
    errxi0=blas[:,2] 
    dd   = blas[:,3]
    dr   = blas[:,4]
    rr   = blas[:,5]

    f=open(filename)
    a=f.readline()
    ng=float(a.split("=")[1].split(" ")[0])
    nr=float(a.split("=")[2].split(" ")[0])
    f.close()

    if normalize is True:
        dd=dd/(ng*(ng-1.)/2.) 
        rr=rr/(nr*(nr-1.)/2.) 
        dr=dr/(ng*nr)         

    return r,xi0,dd,rr,dr,ng,nr

def conv3D2PerpPara(q3D_rm,n_mu,xi0Bool=False):
    ''' 
        xi0Bool: True : xi(r) 
        xi0Bool: False: array( xi(r_perp) ,xi(r_para) )
    '''
    q_perp = sum(q3D_rm[:,:n_mu/2],axis=1)*0.5
    q_para = sum(q3D_rm[:,n_mu/2:],axis=1)*0.5
    
    if xi0Bool: return sum(q3D_rm[:],axis=1)
    else:       return np.concatenate((q_perp,q_para))

def read_cute_rm(filename,n_r=40,n_mu=10,normalize=True,xi0Bool=False,Feature2D=False):
    blas = np.loadtxt(filename,skiprows=1)
    
    mu        = blas[:n_mu,0] 
    r_regular = blas[0:n_r*n_mu:n_mu,1] 
    xi3D_rm   = blas[:,2].reshape(n_r,n_mu)
    errxi3D_rm= blas[:,3].reshape(n_r,n_mu) 
    dd3D_rm   = blas[:,4].reshape(n_r,n_mu)
    dr3D_rm   = blas[:,5].reshape(n_r,n_mu)
    rr3D_rm   = blas[:,6].reshape(n_r,n_mu)

    f=open(filename)
    a=f.readline()
    ng=float(a.split("=")[1].split(" ")[0])
    nr=float(a.split("=")[2].split(" ")[0])
    f.close()

    if normalize is True:
        dd3D_rm=dd3D_rm /(ng*(ng-1.)/2.) 
        rr3D_rm=rr3D_rm /(nr*(nr-1.)/2.) 
        dr3D_rm=dr3D_rm /(ng*nr)         
    if Feature2D:
        r = np.array([r_regular,mu])
    else: 
        xi = conv3D2PerpPara(xi3D_rm,n_mu,xi0Bool=xi0Bool)
        dd = conv3D2PerpPara(dd3D_rm,n_mu,xi0Bool=xi0Bool)
        rr = conv3D2PerpPara(rr3D_rm,n_mu,xi0Bool=xi0Bool)
        dr = conv3D2PerpPara(dr3D_rm,n_mu,xi0Bool=xi0Bool)

        if xi0Bool:  r = r_regular
        else:        r = np.concatenate((r_regular,r_regular))
    
    return r,xi,dd,rr,dr,ng,n_r

def read_cute_combine_regions(file1,file2,normalize=True):
    r1,xi01,dd1,rr1,dr1,ng1,nr1=read_cute(file1,normalize=False)
    r2,xi02,dd2,rr2,dr2,ng2,nr2=read_cute(file2,normalize=False)
    ng=ng1+ng2
    nr=nr1+nr2
    dd=dd1+dd2
    rr=rr1+rr2
    dr=dr1+dr2

    xi0 = 0.5*(xi01+xi02)

    if normalize is True:
        dd=dd/(ng*(ng-1.)/2.) #=dd/(ng*(ng-1)/2) (on nersc when I use the /2 I got false correlation function)
        rr=rr/(nr*(nr-1.)/2.) #=rr/(nr*(nr-1)/2)
        dr=dr/(ng*nr)
    stops
    return r1,xi0,dd,rr,dr,ng,nr


def read_cute_many_backup(files,who_Nest='JC',who_xi='ls',calc_xi=False):
    r,xi0,dd,rr,dr,ng,nr = read_cute(files[0])
    nsim  = len(files)
    nbins = len(r)
    
    all_xi  =np.zeros((nsim,nbins))
    all_nr  =np.zeros((nsim,nbins))
    all_d2  =np.zeros((nsim,nbins))

    all_rhxi  =np.zeros((nsim))
    all_rhnr  =np.zeros((nsim))
    all_rhd2  =np.zeros((nsim))

    counter = 0
    for file_i in files:
        r,all_xi[counter],dd,rr,dr,ng,nr = read_cute(file_i)
        if calc_xi==True: all_xi[counter]=xi(dd,rr,dr,who_xi=who_xi)
        all_nr[counter]  = nr_gal(r,dd,rr,dr,who_Nest=who_Nest,correct_smallscales=True) #nr_gal(r,dd,rr,dr,who_Nest=who_Nest,correct_smallscales=True) #N_of_xi(r,all_xi[counter])
        #if counter==3: stop
        all_d2[counter]  = d2_of_N(r,all_nr[counter],gradient=True) # Last D2recon: False 30May2016
        
        all_rhxi[counter]=rhomo_nr(r,all_xi[counter])

        all_rhnr[counter]=rhomo_nr(r,all_nr[counter])
        
        all_rhd2[counter]=rhomo_d2(r,all_d2[counter])

        counter += 1
        #if counter==98: stop
    print nsim
    mean_xi,sig_xi,covmat_xi,cormat_xi=average_realisations(all_xi)
    mean_nr,sig_nr,covmat_nr,cormat_nr=average_realisations(all_nr)
    mean_d2,sig_d2,covmat_d2,cormat_d2=average_realisations(all_d2)

    return (r,mean_xi,sig_xi,
              mean_nr,sig_nr,
              mean_d2,sig_d2,
              covmat_xi,covmat_nr,covmat_d2,
              cormat_xi,cormat_nr,cormat_d2,
              all_rhxi,all_rhnr,all_rhd2,
              all_xi,all_nr,all_d2)

def read_cute_many(files,
    cosmoFID={},cosmoGAU={},biasFID=None,sigpFID=None,zmid=0.5,kaiserFID=False,dampingFID=False,galaxyFID=False,biasGAU=2.0,sigpGAU=350.,kaiserGAU=False,dampingGAU=False,galaxyGAU=False,change_gauge=True,change_gauge_xi=False,
    who_Nest='JC',who_xi='ls',calc_xi=False):
    
    r,xi0,dd,rr,dr,ng,nr = read_cute(files[0])
    nsim  = len(files)
    nbins = len(r)
    
    all_xi  =np.zeros((nsim,nbins))
    all_nr  =np.zeros((nsim,nbins))
    all_d2  =np.zeros((nsim,nbins))
    all_rhxi  =np.zeros((nsim))
    all_rhnr  =np.zeros((nsim))
    all_rhd2  =np.zeros((nsim))
    
    counter = 0
    for file_i in files:
        r,all_xi[counter],dd,rr,dr,ng,nr = read_cute(file_i)
        all_nr[counter]  = N_of_xi(r,all_xi[counter])
        if calc_xi: all_xi[counter]=xi(dd,rr,dr,who_xi=who_xi)
        if calc_xi: all_nr[counter]  = nr_gal(r,dd,rr,dr,who_Nest=who_Nest,correct_smallscales=True) #nr_gal(r,dd,rr,dr,who_Nest=who_Nest,correct_smallscales=True) #N_of_xi(r,all_xi[counter])

        #if counter==3: stop

        all_rhxi[counter]=rhomo_nr(r,all_xi[counter])

        counter += 1            

    if change_gauge_xi:
        r_gal2MMgauge_xi,all_xi,alphas_gal2gauge_xi = obsToMMgauge(r,all_xi,cosmoFID=cosmoFID,cosmoGAU=cosmoGAU,biasFID=biasFID,sigpFID=sigpFID,zmid=zmid,kaiserFID=kaiserFID,dampingFID=dampingFID,galaxyFID=galaxyFID,who_give='xi',biasGAU=biasGAU,sigpGAU=sigpGAU,kaiserGAU=kaiserGAU,dampingGAU=dampingGAU,galaxyGAU=galaxyGAU) # since kaiserGAU=dampingGAU=False then GAU

    if change_gauge:
        r_gal2MMgauge,all_nr_gal2MMgauge,alphas_gal2gauge = obsToMMgauge(r,all_nr,cosmoFID=cosmoFID,cosmoGAU=cosmoGAU,biasFID=biasFID,sigpFID=sigpFID,zmid=zmid,kaiserFID=kaiserFID,dampingFID=dampingFID,galaxyFID=galaxyFID,who_give='nr',biasGAU=biasGAU,sigpGAU=sigpGAU,kaiserGAU=kaiserGAU,dampingGAU=dampingGAU,galaxyGAU=galaxyGAU) # since kaiserGAU=dampingGAU=False then GAU=MM
    else:
        all_nr_gal2MMgauge = all_nr

    for i in range(len(files)):

        all_nr[i,:]=all_nr_gal2MMgauge[i,:]
        all_rhnr[i]=rhomo_nr(r,all_nr[i,:])
        thed2 = d2_of_N(r,all_nr[i,:])

        all_d2[i,:]=thed2
        all_rhd2[i]=rhomo_d2(r,thed2)  #        all_rhd2[i]=rhomo_d2(r[1:],thed2)

    #print nsim
    mean_xi,sig_xi,covmat_xi,cormat_xi=average_realisations(all_xi)
    mean_nr,sig_nr,covmat_nr,cormat_nr=average_realisations(all_nr)
    mean_d2,sig_d2,covmat_d2,cormat_d2=average_realisations(all_d2)
    
    print calc_xi
    return (r,mean_xi,sig_xi,
              mean_nr,sig_nr,
              mean_d2,sig_d2,
              covmat_xi,covmat_nr,covmat_d2,
              cormat_xi,cormat_nr,cormat_d2,
              all_rhxi,all_rhnr,all_rhd2,
              all_xi,all_nr,all_d2)

############################ Shannon Entropy Homogeneity ####################################################

def seh(dd,ddlogdd,ng,Normalize=True):
    ''' 
    #HH = 1. - ( ddlogdd - dd*np.log10(ng*(ng-1)) ) / ( dd*( np.log10(dd) - np.log10( ng*(ng-1) ) ) ) 
    '''"""
    if Normalize: Norm=dd*np.log10(ng*(ng-1))
    else: Norm=0.0
    HH = 1. - ( ddlogdd -Norm )/( dd*np.log10(dd) -Norm)
    """
    #or 
    
    Hr = np.log10(dd) - ddlogdd/dd 
    if Normalize : HH = Hr/np.log10(ng)
    else:  HH = Hr/np.log10(10.0)
    return HH

def slope_seh(r,dd,rr,ddlogdd,rrlogrr,ng,nr,AAA=1.,NormSEH=False):
    HHD = seh(dd,ddlogdd,ng,Normalize=NormSEH)
    HHR = seh(rr,rrlogrr,nr,Normalize=NormSEH)
    ratio =  HHD/HHR # (1.-HHD)/(1.-HHR) 
    slope = np.gradient( np.log(ratio) , np.gradient(np.log(r)) ) + AAA
    #xi = ratio - 1.
    #xi_r2 = xi*r**2
    #nr = 1. + integrate.cumtrapz(xi_r2,x=r,initial=xi_r2[0])*3/r**3
    #slope = np.gradient( np.log(nr) , np.gradient(np.log(r)) ) + 3.
    return ratio,slope,HHD,HHR

def seh_ls(dd,rr,dr,ng,nr):
    dddd = ( (nr*np.log10(nr)/(ng*np.log10(ng)) )*( dd*np.log10(dd)/(rr*np.log10(rr)) ) ) - ( (nr*np.log10(nr)/(ng*np.log10(ng)) )*( dr*np.log10(dr)/(rr*np.log10(rr)) ) )  

def slope_seh_ls(r,dd,rr,dr,ng,nr):
    xi_ls = seh_ls(dd,rr,dr,ng,nr)
    xi_ls_r2 = xi_ls*r**2
    nr = 1. + integrate.cumtrapz(xi_ls_r2,x=r,initial=xi_ls_r2[0])*3/r**3
    d2 = np.gradient( np.log(nr) , np.gradient(np.log(r)) ) + 3.
    return xi_ls,nr,d2

def read_cute_se(filename,normalize=True,give_rightedge=False):
    blas = np.loadtxt(filename,skiprows=1)
    if give_rightedge: addReachRedge=0.5*(1500./50.) 
    else: addReachRedge=0.0
    r    = blas[:,0] + addReachRedge
    xi0  = blas[:,1]
    errxi0=blas[:,2] 
    dd   = blas[:,3]
    dr   = blas[:,4]
    rr   = blas[:,5]
    ddlogdd = blas[:,6]
    rrlogrr = blas[:,7]
    drlogdr = blas[:,8]

    f=open(filename)
    a=f.readline()
    ng=float(a.split("=")[1].split(" ")[0])
    nr=float(a.split("=")[2].split(" ")[0])
    f.close()

    if normalize is True:
        dd=dd/(ng*(ng-1.)/2.) 
        rr=rr/(nr*(nr-1.)/2.) 
        dr=dr/(ng*nr)         

    return r,xi0,dd,rr,dr,ng,nr,ddlogdd,rrlogrr,drlogdr

def read_cute_many_se(files,
    cosmoFID={},cosmoGAU={},biasFID=None,sigpFID=None,zmid=0.5,kaiserFID=False,dampingFID=False,galaxyFID=False,biasGAU=2.0,sigpGAU=350.,kaiserGAU=False,dampingGAU=False,galaxyGAU=False,change_gauge=True,
    who_Nest='JC',who_xi='ls',calc_xi=False):

    r,xi0,dd,rr,dr,ng,nr,ddlogdd,rrlogrr,drlogdr = read_cute_se(files[0],normalize=False)
    nsim  = len(files)
    nbins = len(r)

    all_xi   =np.zeros((nsim,nbins))    
    all_sed  =np.zeros((nsim,nbins))
    all_ser  =np.zeros((nsim,nbins))
    all_se   =np.zeros((nsim,nbins))
    all_ss   =np.zeros((nsim,nbins))
    all_rhxi  =np.zeros((nsim))
    all_rhse  =np.zeros((nsim))

    counter = 0
    for file_i in files:
        r,all_xi[counter],dd,rr,dr,ng,nr,ddlogdd,rrlogrr,drlogdr = read_cute_se(file_i,normalize=False)
        all_xi[counter]=xi(dd/(ng*(ng-1.)),rr/(nr*(nr-1.)),dr/(ng*nr),who_xi='ph')
        all_se[counter],all_ss[counter],all_sed[counter],all_ser[counter]=slope_seh(r,dd,rr,ddlogdd,rrlogrr,ng,nr,NormSEH=False)

        #all_xi[counter],all_se[counter],all_ss[counter]=slope_seh_ls(r,dd,rr,dr,ng,nr)
        
        all_rhxi[counter]=rhomo_nr(r,all_xi[counter])
        counter += 1            

    if change_gauge:
        r_gal2MMgauge,all_nr_gal2MMgauge,alphas_gal2gauge = obsToMMgauge(r,all_se,cosmoFID=cosmoFID,cosmoGAU=cosmoGAU,biasFID=biasFID,sigpFID=sigpFID,zmid=zmid,kaiserFID=kaiserFID,dampingFID=dampingFID,galaxyFID=galaxyFID,who_give='nr',biasGAU=biasGAU,sigpGAU=sigpGAU,kaiserGAU=kaiserGAU,dampingGAU=dampingGAU,galaxyGAU=galaxyGAU) # since kaiserGAU=dampingGAU=False then GAU=MM
    else:
        all_se_gal2MMgauge = all_se

    mean_xi,sig_xi,covmat_xi,cormat_xi=average_realisations(all_xi)
    mean_se,sig_se,covmat_se,cormat_se=average_realisations(all_se)
    mean_ss,sig_ss,covmat_ss,cormat_ss=average_realisations(all_ss)
    mean_sed,sig_sed,covmat_sed,cormat_sed=average_realisations(all_sed)
    mean_ser,sig_ser,covmat_ser,cormat_ser=average_realisations(all_ser)
    

    stop # errorbar(r,mean_se,yerr=np.sqrt(np.diag(covmat_se)))
    return (r,mean_xi,sig_xi,
              mean_se,sig_se,
              mean_ss,sig_ss,
              covmat_xi,covmat_se,covmat_ss,
              cormat_xi,cormat_se,cormat_ss,
              all_rhxi,all_rhse,all_rhss,
              all_xi,all_se,all_ss)

############################ END: Shannon Entropy Homogeneity ####################################################

def read_cute_many_rm(files,
    cosmoFID={},cosmoGAU={},biasFID=None,sigpFID=None,zmid=0.5,kaiserFID=False,dampingFID=False,galaxyFID=False,biasGAU=2.0,sigpGAU=350.,kaiserGAU=False,dampingGAU=False,galaxyGAU=False,change_gauge=True,
    who_Nest='JC',who_xi='ls',xi0Bool=True,calc_xi=True,corr_lss=True):
    # if calc_xi : xi from my function instead cute
    # if corr_lss: puts xi(r_lss)=0 where cute gives rr(r_lss)=0
    
    r,xi0,dd,rr,dr,ng,n_r = read_cute_rm(files[0],xi0Bool=xi0Bool)
    nsim  = len(files)
    nbins = len(r)

    all_xi  =np.zeros((nsim,nbins))
    all_nr  =np.zeros((nsim,nbins))
    all_d2  =np.zeros((nsim,nbins))
    if xi0Bool:
        all_rhxi,all_rhnr,all_rhd2  =np.zeros((nsim)),np.zeros((nsim)),np.zeros((nsim))
    else: 
        all_rhxi,all_rhnr,all_rhd2  =np.zeros((nsim,2)),np.zeros((nsim,2)),np.zeros((nsim,2))

    counter = 0
    for file_i in files:
        r,all_xi[counter],dd,rr,dr,ng,n_r = read_cute_rm(file_i,xi0Bool=xi0Bool)
        all_nr[counter]  = N_of_xi(r,all_xi[counter])
        if calc_xi: 
            if xi0Bool: all_xi[counter]=xi(dd,rr,dr,who_xi=who_xi,corr_lss=corr_lss)
            else:       
                all_xi[counter][:n_r]=xi(dd[:n_r],rr[:n_r],dr[:n_r],who_xi=who_xi,corr_lss=corr_lss)
                all_xi[counter][-n_r:]=xi(dd[-n_r:],rr[-n_r:],dr[-n_r:],who_xi=who_xi,corr_lss=corr_lss)
        if calc_xi: 
            if xi0Bool: all_nr[counter] = nr_gal(r,dd,rr,dr,who_Nest=who_Nest,correct_smallscales=False,corr_lss=corr_lss) #nr_gal(r,dd,rr,dr,who_Nest=who_Nest,correct_smallscales=True) #N_of_xi(r,all_xi[counter])
            else: 
                all_nr[counter][:n_r]  = nr_gal(r[:n_r],dd[:n_r],rr[:n_r],dr[:n_r],who_Nest=who_Nest,correct_smallscales=False,corr_lss=corr_lss) #nr_gal(r,dd,rr,dr,who_Nest=who_Nest,correct_smallscales=True) #N_of_xi(r,all_xi[counter])
                all_nr[counter][-n_r:]  = nr_gal(r[-n_r:],dd[-n_r:],rr[-n_r:],dr[-n_r:],who_Nest=who_Nest,correct_smallscales=False,corr_lss=corr_lss) #nr_gal(r,dd,rr,dr,who_Nest=who_Nest,correct_smallscales=True) #N_of_xi(r,all_xi[counter])
        
        #if counter==3: stop

        if xi0Bool:
            all_rhxi[counter]=rhomo_nr(r,all_xi[counter])
        else:
            all_rhxi[counter][0]=rhomo_nr(r[:n_r],all_xi[counter][:n_r])
            all_rhxi[counter][1]=rhomo_nr(r[:n_r],all_xi[counter][-n_r:])

        counter += 1        

    if change_gauge:
        r_gal2MMgauge,all_nr_gal2MMgauge,alphas_gal2gauge = obsToMMgauge(r,all_nr,cosmoFID=cosmoFID,cosmoGAU=cosmoGAU,biasFID=biasFID,sigpFID=sigpFID,zmid=zmid,kaiserFID=kaiserFID,dampingFID=dampingFID,galaxyFID=galaxyFID,who_give='nr',biasGAU=biasGAU,sigpGAU=sigpGAU,kaiserGAU=kaiserGAU,dampingGAU=dampingGAU,galaxyGAU=galaxyGAU) # since kaiserGAU=dampingGAU=False then GAU=MM
    else:
        all_nr_gal2MMgauge = all_nr

    for i in range(len(files)):

        all_nr[i,:]=all_nr_gal2MMgauge[i,:]
        if xi0Bool: 
            all_rhnr[i]=rhomo_nr(r,all_nr[i,:])
        else: 
            all_rhnr[i][0]=rhomo_nr(r[:n_r],all_nr[i,:][:n_r])
            all_rhnr[i][1]=rhomo_nr(r[:n_r],all_nr[i,:][-n_r:])
        thed2 = d2_of_N(r,all_nr[i,:])

        all_d2[i,:]=thed2
        if xi0Bool: all_rhd2[i]=rhomo_d2(r,thed2) 
        else: 
            all_rhd2[i][0]=rhomo_d2(r[:n_r],thed2[:n_r])  #        all_rhd2[i]=rhomo_d2(r[1:],thed2)
            all_rhd2[i][1]=rhomo_d2(r[:n_r],thed2[-n_r:])

    #print nsim
    mean_xi,sig_xi,covmat_xi,cormat_xi=average_realisations(all_xi)
    mean_nr,sig_nr,covmat_nr,cormat_nr=average_realisations(all_nr)
    mean_d2,sig_d2,covmat_d2,cormat_d2=average_realisations(all_d2)
    
    #print calc_xi
    return (r,mean_xi,sig_xi,
              mean_nr,sig_nr,
              mean_d2,sig_d2,
              covmat_xi,covmat_nr,covmat_d2,
              cormat_xi,cormat_nr,cormat_d2,
              all_rhxi,all_rhnr,all_rhd2,
              all_xi,all_nr,all_d2)

def read_cute_many_combine(filesN,filesS,
    cosmoFID={},cosmoGAU={},biasFID=None,sigpFID=None,zmid=0.5,kaiserFID=False,dampingFID=False,galaxyFID=False,biasGAU=2.0,sigpGAU=350.,kaiserGAU=False,dampingGAU=False,galaxyGAU=False,change_gauge=True,
    who_Nest='JC',who_xi='ls',calc_xi=False):

    r,xi0,dd,rr,dr,ng,nr = read_cute(filesN[0],normalize=True)
    nsim  = len(filesN)
    nbins = len(r)
    
    all_xi  =np.zeros((nsim,nbins))
    all_nr  =np.zeros((nsim,nbins))
    all_d2  =np.zeros((nsim,nbins))
    all_rhxi  =np.zeros((nsim))
    all_rhnr  =np.zeros((nsim))
    all_rhd2  =np.zeros((nsim))

    counter = 0
    for fileN_i,fileS_i in zip(filesN,filesS):
        r,all_xi[counter],dd,rr,dr,ng,nr = read_cute_combine_regions(fileN_i,fileS_i)
        all_nr[counter]  = N_of_xi(r,all_xi[counter])
        if calc_xi: all_xi[counter]=xi(dd,rr,dr,who_xi=who_xi)
        if calc_xi: all_nr[counter]  = nr_gal(r,dd,rr,dr,who_Nest=who_Nest,correct_smallscales=True) #nr_gal(r,dd,rr,dr,who_Nest=who_Nest,correct_smallscales=True) #N_of_xi(r,all_xi[counter])

        #if counter==3: stop

        all_rhxi[counter]=rhomo_nr(r,all_xi[counter])

        counter += 1        

    if change_gauge:
        r_gal2MMgauge,all_nr_gal2MMgauge,alphas_gal2gauge = obsToMMgauge(r,all_nr,cosmoFID=cosmoFID,cosmoGAU=cosmoGAU,biasFID=biasFID,sigpFID=sigpFID,zmid=zmid,kaiserFID=kaiserFID,dampingFID=dampingFID,galaxyFID=galaxyFID,who_give='nr',biasGAU=biasGAU,sigpGAU=sigpGAU,kaiserGAU=kaiserGAU,dampingGAU=dampingGAU,galaxyGAU=galaxyGAU) # since kaiserGAU=dampingGAU=False then GAU=MM
    else:
        all_nr_gal2MMgauge = all_nr

    for i in range(len(filesN)):

        all_nr[i,:]=all_nr_gal2MMgauge[i,:]
        all_rhnr[i]=rhomo_nr(r,all_nr[i,:])
        thed2 = d2_of_N(r,all_nr[i,:])

        all_d2[i,:]=thed2
        all_rhd2[i]=rhomo_d2(r,thed2)  #        all_rhd2[i]=rhomo_d2(r[1:],thed2)

    #print nsim
    mean_xi,sig_xi,covmat_xi,cormat_xi=average_realisations(all_xi)
    mean_nr,sig_nr,covmat_nr,cormat_nr=average_realisations(all_nr)
    mean_d2,sig_d2,covmat_d2,cormat_d2=average_realisations(all_d2)
    
    print calc_xi

    return (r,mean_xi,sig_xi,
              mean_nr,sig_nr,
              mean_d2,sig_d2,
              covmat_xi,covmat_nr,covmat_d2,
              cormat_xi,cormat_nr,cormat_d2,
              all_rhxi,all_rhnr,all_rhd2,
              all_xi,all_nr,all_d2)

def get_cute_obs(files,who_give='d2',who_Nest='JC',calc_xi=False,change_gauge=False):
    r,mean_xi,sig_xi,mean_nr,sig_nr,mean_d2,sig_d2,covmat_xi,covmat_nr,covmat_d2,cormat_xi,cormat_nr,cormat_d2,all_rhxi,all_rhnr,all_rhd2,all_xi,all_nr,all_d2 = read_cute_many(files,who_Nest=who_Nest,calc_xi=calc_xi,change_gauge=change_gauge)
    if who_give=='xi':
        return (r,mean_xi,sig_xi,covmat_xi,cormat_xi,all_xi)
    if who_give=='nr':
        return (r,mean_nr,sig_nr,covmat_nr,cormat_nr,all_nr)
    if who_give=='d2':
        return (r,mean_d2,sig_d2,covmat_d2,cormat_d2,all_d2)

### END: PAIRS CUTE

def mean_all_scale(nFractal,fname='fractal_xi0_%0.3f_*.dat',rmin=40.,who_Nest='JC'):
    fractal_dir = '/Users/pntelis/Python/BOSS/LAURENT_FRACTALS/FRACTAL_ANALYSIS/FitsFiles/match_frac2rands/Cute/FRACTAL_pts_LAST/'
    #'BACKUP_10000pts_xi0_logr/' # 'FRACTAL_pts_LAST/'
    frFiles_25 = glob.glob(fractal_dir+fname%nFractal)
    res     = get_cute_obs(frFiles_25,who_give='d2',who_Nest=who_Nest,calc_xi=True)
    wok = np.where(res[0]>rmin)
    data = res[1][wok]
    dim_fractal = np.mean( data )
    errdim_fractal=  np.std( res[1][wok])/np.sqrt(len(data) )
    return dim_fractal,errdim_fractal

def compute_mean_fractal(nFractal=np.array([2.950,2.970,2.990,2.995,2.999,3.000]),who_Nest=['str','JC','JM'],color=['b','g','r'],d2label=['str','cor','lau'],rmin=15.,doFit=False):
    ''' Usage: galtools.compute_mean_fractal(nFractal=np.array([2.950,2.970,2.990,2.995,3.000]),who_Nest=['JC'],color=['g'],d2label=['cor'],rmin=15.,doFit=True) '''
    dim_fractal = np.zeros(len(nFractal))
    errdim_fractal = np.zeros(len(nFractal))
    Dmove=0.0
    ARRwho_Nest=array(who_Nest) # Variable that helps to put legend or not
    for who_Nest,color,d2lab_i in zip(ARRwho_Nest,color,d2label):

        for i in xrange(len(nFractal)) :            
            dim_fractal[i],errdim_fractal[i] = mean_all_scale(nFractal[i],rmin=rmin,who_Nest=who_Nest)
        
        print nFractal
        print dim_fractal
        print errdim_fractal
        print errdim_fractal/dim_fractal
        #Dmove += 0.0003
        if doFit:
            def straight(x,pars):
                return pars[0]*x+pars[1]
            res = fitting.dothefit(nFractal,dim_fractal,errdim_fractal,[1.0,0.1],functname=straight)
            d2lab_i = d2lab_i+'\n a=%0.1e$\pm$%0.1e \n b=%0.1e$\pm$%0.1e \n $\chi^2$=%0.1e/%0.0f'%(res[1][0],res[2][0],res[1][1],res[2][1],res[4],res[5])
        errorbar(nFractal+Dmove,dim_fractal/nFractal-1.,yerr=errdim_fractal/nFractal,fmt=color+'o',label=d2lab_i)
        x = np.linspace(np.min(nFractal)*0.99,np.max(nFractal)*1.01)
        plot(x,x*0.0,'k--')
        xlim(min(nFractal)-0.01,max(nFractal)+ .01)
        ylim(-0.002,+0.002)
        if(shape(ARRwho_Nest)[0]>1): legend(loc=2,numpoints=1,frameon=False)
        ylabel('$\mathcal{D}^{recon}_{2}/\mathcal{D}_{2}^{input}-1$',size=20),xlabel('$\mathcal{D}_{2}^{input}$',size=20)
        draw()

### END: FRACTALS ANALYSIS AWESOME DAY!!

# COSMO PARAMETERS MEASURE

def fitParsLoad(zi=0,QPM2PL_a=1.0,randVal=0.0,xstart=40.,xstop=1300.,effnum=10.0,who_give='d2',who_Nest='JC',nbspl=5,doplot=False,):
    
    zedge,znames,dz,zmid,zbins = NameZ2(minz=0.43,maxz=0.7,nbins=5)
    rfit   = [1.0,40.0]
    eff      = '_eff_'+str(effnum) #      ''  ,  '_eff_1.5'
    #!#who_give = sys.argv[2]      # 'd2' d2  nr nr
    #!#who_Nest = sys.argv[3]      # 'JC' JM  JC JM Last need 40-150
    #!#nbspl    = int(sys.argv[4]) #  7       6  8 

    if   who_give=='d2': rFitSpl = [40.,100] # [100.,600.] #[30.,150] 
    elif who_give=='nr': rFitSpl = [40.,100] # [100.,600.] #

    xxbias,which_methbias  = 'bias'   , '_xibias'
    method_fitbias = 'minuit' # 'mcmc' , 'minuit'

    addtoname = '_multi'+which_methbias+'_'+method_fitbias+eff+'_'+str(rfit[0])+'-'+str(rfit[1]) #
    load_path = '/Users/pntelis/Python/BOSS/DR12Analysis/Runners12/NPZsFiles/DM_RH_mesure/'
    DM_RH_mesure = np.load(load_path+'DM_RH_measure_'+who_give+'_'+who_Nest+'_'+str(nbspl)+addtoname+'_r_'+str(rFitSpl[0])+'-'+str(rFitSpl[1])+'.npz')

    covmatObsS = DM_RH_mesure['covmatObsS']
    covmatObsN = DM_RH_mesure['covmatObsN'] 
    rS = DM_RH_mesure['rS']
    rN = DM_RH_mesure['rN']
    ObsS = DM_RH_mesure['ObsS']
    ObsN = DM_RH_mesure['ObsN']

    cosmo_Planck = pk_Class.cosmo_Planck(halofit=True)

    pkTheory = pk_Class.theory(cosmo_Planck,zmid[zi])

    def functname(x,pars):
        xxx =  x*QPM2PL_a*cosmology.D_V_simple(zmid[zi],omegam=pars[0],omegax=pars[1],h=pars[2])/cosmology.D_V_simple(zmid[zi],omegam=0.3156,omegax=1-0.3156,h=0.6727)
        res = pkTheory.d2(xxx)#,bias=pars[3],sigp=pars[4],kaiser=kaiser,damping=damping)
        return res

    x,y,cov=rN,ObsS[zi],covmatObsS[zi]+np.diag(np.random.rand(len(rN))*randVal)

    w=np.where((x >= xstart) & (x <= xstop))
    yerr=np.sqrt(np.diag(cov))
    thex=x[w]
    they=y[w]
    theyerr=np.sqrt(cov[w[0],w[0]])
    thecov=(cov[w[0],:])[:,w[0]]

    res = fitting.dothefit(thex,they,thecov,[0.3175,1-0.3175,0.6727],parbounds=[(0.25,0.35),(0.65,0.75),(0.6,0.7),],functname=functname,method='minuit',nmock_prec=1000)

    for i in np.arange(len(res[1])): print res[1][i].round(3) , res[2][i].round(3) , (res[2][i]/res[1][i]).round(3)

    if doplot:
        
        plt.figure(1),plt.clf()
        plt.subplot(1,2,1),plt.yscale('log'),plt.xscale('log')
        plt.plot(x,np.diagonal(cov),'r')
        plt.plot(x,np.diagonal( covmatObsS[zi] ),'b')

        plt.figure(1),
        plt.subplot(1,2,2),plt.yscale('log'),plt.xscale('log')
        plt.plot(thex,np.diagonal(thecov),'r')
        plt.plot(thex,np.diagonal( (covmatObsS[zi][w[0],:])[:,w[0]] ),'b')

        plt.figure(2),plt.clf()
        plt.subplot(1,2,1),plt.yscale('log'),plt.xscale('log')
        plt.errorbar(x,y,yerr=np.sqrt(np.diagonal(cov)),fmt='r.-')
        plt.errorbar(x,y,yerr=np.sqrt(np.diagonal( covmatObsS[zi] )),fmt='b.-')

        plt.figure(2),
        plt.subplot(1,2,2),plt.yscale('log'),plt.xscale('log')
        plt.errorbar(thex,they,yerr=np.sqrt(np.diagonal(thecov)) ,fmt='r.-')
        plt.errorbar(thex,they,yerr=np.sqrt(np.diagonal( (covmatObsS[zi][w[0],:])[:,w[0]] )) ,fmt='b.-')


### SUFFLING PROJECT 

def err_photo_z(z,sig=0.001):
    ''' 
    Photo-z errors from SnIa calibrations
    https://arxiv.org/pdf/astro-ph/0609591v1.pdf 
    equation 2.4 
    '''
    return sig*(1+z)

def suffle_z(z,sig=0.001):
    z_new = np.random.normal(loc=z,scale=err_photo_z(z,sig=sig))
    return z_new

### END: SUFFLING PROJECT 

def cute_runner():
    print 'IMPLEMENT me'
    workdir = '/Users/pntelis/CUTE/CUTE-1.3/CUTE/'
    filename = './param_cute.ini'
    f = open(filename,'r+')
    text = f.read()
    text = re.sub('qpmDATA_[\d,.,\d]*_[0-9]*[0-9]*[0-9]*[0-9]*_zdr.dat','qpmDATA_'+format_nFractal_nSeed%(nFractal,nSeed)+'_zdr.dat',text)
    text = re.sub('qpmRAND_[\d,.,\d]*_[0-9]*[0-9]*[0-9]*[0-9]*_zdr.dat','qpmRAND_'+format_nFractal_nSeed%(nFractal,nSeed)+'_zdr.dat',text)
    text = re.sub('qpm_xi0_[\d,.,\d]*_[0-9]*[0-9]*[0-9]*[0-9]*.dat','qpm_xi0_'+format_nFractal_nSeed%(nFractal,nSeed)+'.dat',text)
    f.seek(0)
    f.write(text)
    f.truncate()
    ##print text
    f.close()
    #!#subprocess.call([workdir+'./CUTE',filename])



##### For fast integrations!!!

def produceNPZ(whichAreSaved,savefile,who_give,who_Nest,xgc,nmock=1000,change_gauge=False):
    ''' IMPLEMENT ME'''
    zedge,znames,dz,zmid,zbins= NameZ2(minz=0.43,maxz=0.7,nbins=5) ### SOS nbins=5
    datafileNorth=[]
    datafileSouth=[]
    mockdirNorth=[]
    mockdirSouth=[]
    for i in znames:
        datafileNorth.append('/users/pntelis/Python/BOSS/DR12Analysis/WeightPairs12_Nersc/dr12North_'+np.str(i)+'_pairs_weighted.txt')
        datafileSouth.append('/Users/pntelis/Python/BOSS/DR12Analysis/WeightPairs12_Nersc/dr12South_'+np.str(i)+'_pairs_weighted.txt')
        mockdirNorth.append('/Volumes/Data/Pierros/SDSS/DR12/MockQPMs_ALL_NERSC/CountPairs_log/'+np.str(i)+'/ngc/')
        mockdirSouth.append('/Volumes/Data/Pierros/SDSS/DR12/MockQPMs_ALL_NERSC/CountPairs_log/'+np.str(i)+'/sgc/')

    pairfiles = "pairs_mock_*.txt"

    if  xgc=='sgc':
        mockdir = mockdirSouth
        datafile= datafileSouth
    elif xgc=='ngc':
        mockdir = mockdirNorth
        datafile= datafileNorth
    else:
        print 'put correct galactic cap'

    dimData=51
    Obs=zeros((zbins,dimData))
    covmatObs = zeros((zbins,dimData,dimData))
    allObs    = zeros((zbins,nmock,dimData))
    rhAllObs  = zeros((zbins,nmock))
    meanObs   = zeros((zbins,dimData))

    res_fit_z = []

    for i in np.arange(np.size(znames)): 

        r_data,d2_data,nr_data,xi_data=read_data(datafile[i],who_xi='ls',who_Nest=who_Nest)
        if who_give=='d2': Obs[i]=d2_data
        elif who_give=='nr': Obs[i]=nr_data
        elif who_give=='xi': Obs[i]=xi_data

        covmatObs[i],meanObs[i],rhAllObs[i],allObs[i],r,nmock_prec=read_mocks(mockdir[i],change_gauge=change_gauge,who_give=who_give,who_Nest=who_Nest)

    savez(savefile,Obs=Obs,covmatObs=covmatObs,meanObs=meanObs,rhAllObs=rhAllObs,allObs=allObs,r=r,nmock_prec=nmock_prec)
    file=open(whichAreSaved,'rw')
    file.write(savefile+" \n")
    file.close()
