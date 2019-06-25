import numpy as np
from pylab import *
from numpy import *
#import mpfit
import iminuit
#import emcee
# Taken from J.C.Hamilton and remodified by P.Ntelis June 2014

####### Generic polynomial function ##########
def thepolynomial(x,pars):
    f=np.poly1d(pars)
    return(f(x))

####### Generic fitting function ##############
def dothefit(x,y,covarin,guess,functname=thepolynomial,parbounds=None,nmock_prec=None,method='minuit'):
    if method == 'minuit':
        print('Fitting with Minuit')
        return(do_minuit(x,y,covarin,guess,functname=functname,parbounds=parbounds,nmock_prec=nmock_prec))
    elif method == 'mpfit':
        print('Fitting with MPFit')
        return(do_mpfit(x,y,covarin,guess,functname=functname,nmock_prec=nmock_prec))
    elif method == 'mcmc':
        print('Fitting with MCMC')
        return(do_emcee(x,y,covarin,guess,functname,nmock_prec=nmock_prec))
    else:
        print('method must be among: minuit, mpfit, mcmc')
        return(0,0,0,0)

################### Fitting with MPFIT #######################
# Function to be minimized returning the right stuff
def chi2svd(pars, fjac=None, x=None, y=None, svdvals=None, v=None, functname=thepolynomial):
    model=functname(x,pars)
    status = 1
    resid=dot(v, y-model)/sqrt(svdvals)
    return([status,resid])

def fdeviates(pars,fjac=None,x=None,y=None,err=None, functname=thepolynomial):
    model=functname(x,pars)
    status=1
    return([status, (y-model)/err])

def do_mpfit(x,y,covarin,guess,functname=thepolynomial,nmock_prec=None):
    # check if covariance or error bars were given
    #covar=covarin
    if nmock_prec!=None: covarin = covarin * (nmock_prec-1.)/(nmock_prec-len(x)-2.)
    if np.size(np.shape(covarin)) == 1:
        err=covarin
        print('err')
        #Prepare arguments for mpfit
        fa={'x':double(x),'y':double(y),'err':double(err),'functname':functname}
        costfct=fdeviates
    else:
        print('covar')
        #First do a SVD decomposition
        u,s,v=np.linalg.svd(covarin)
        #Prepare arguments for mpfit
        fa={'x':double(x),'y':double(y),'svdvals':double(s),'v':double(v),'functname':functname}
        costfct=chi2svd

    #Run MPFIT
    mpf = mpfit.mpfit(costfct, guess, functkw=fa)

    print('Status of the Fit',mpf.status)
    print('Chi2=',mpf.fnorm)
    print('ndf=',mpf.dof)
    print('Fitted params:',mpf.params)

    return(mpf,mpf.params,mpf.perror,mpf.covar, mpf.fnorm, mpf.dof )

################################################################

################### Fitting with Minuit #######################
# Class defining the minimizer and the data
class MyChi2:
    def __init__(self,xin,yin,covarin,functname):
        self.x=xin
        self.y=yin
        self.covar=covarin
        self.invcov=np.linalg.inv(covarin) 
        self.functname=functname

    def __call__(self,*pars):
        val=self.functname(self.x,pars)
        chi2=dot(dot(self.y-val,self.invcov),self.y-val)
        return(chi2)

def do_minuit(x,y,covarin,guess,functname=thepolynomial,nmock_prec=None,parbounds=None):
    # check if covariance or error bars were given
    # input - covarin: either covariance matrix(x,x) or std(x,x)
    if nmock_prec!=None: covarin = covarin * (nmock_prec-1.)/(nmock_prec-len(x)-2.)
    
    covar=covarin
    if np.size(np.shape(covarin)) == 1:
        err=covarin
        covar=np.zeros((np.size(err),np.size(err)))
        covar[np.arange(np.size(err)),np.arange(np.size(err))]=err**2

    # instantiate minimizer
    chi2=MyChi2(x,y,covar,functname)
    # variables
    ndim=np.size(guess)
    parnames =[]
    errnames =[]
    parlimits=[] #!#
    for i in range(ndim): 
        parnames.append('c'+np.str(i))
        errnames.append('error_c'+np.str(i))
        if (parbounds!=None): parlimits.append('limit_c'+np.str(i))    #!#
    
    for truc in parnames: errnames.append(truc)
    if (parbounds!=None): 
        for pragma in parlimits: errnames.append(pragma) #!#
    fract=0.001
    values=(np.array(guess)*fract).tolist()
    for val in guess: values.append(val)
    if (parbounds!=None): 
        for val2 in parbounds: values.append(val2)
    
    # initial guess
    #theguess=dict(zip(parnames,guess))
    theguess=dict(zip(errnames,values))
    
    # Run Minuit
    print('Fitting with Minuit')
    m = iminuit.Minuit(chi2,forced_parameters=parnames,errordef=1.0,**theguess)
    m.migrad()
    #m.migrad()
    #m.hesse()
    # build np.array output
    parfit=[]
    for i in parnames: parfit.append(m.values[i])
    errfit=[]
    for i in parnames: errfit.append(m.errors[i])
    covariance=np.zeros((ndim,ndim))
    try:
        for i in np.arange(ndim):
            for j in np.arange(ndim):
                covariance[i,j]=m.covariance[(parnames[i],parnames[j])]
    except TypeError:
        print( "No accurate covmat was built for params")
        print(" put cov=matrix(0)")
        pass

    print('Chi2=',chi2(*parfit))
    print('ndf=',np.size(x)-ndim)
    return(m,np.array(parfit), np.array(errfit), np.array(covariance), chi2(*parfit), np.size(x)-ndim)

################################################################



################### Fitting with Emcee #######################
# Class defining the minimizer and the data
def lnprobcov(thepars, xvalues, yvalues, svdvals, v, functname):
    resid=dot(v, yvalues-functname(xvalues, thepars))
    chi2=np.sum((resid/np.sqrt(svdvals))**2)
    return(-0.5*chi2)

def do_emcee(x,y,covarin,guess,functname=thepolynomial,nmock_prec=None,nburn=1000,nbmc=4000,nthreads=0):
    # check if covariance or error bars were given
    if nmock_prec!=None: covarin = covarin * (nmock_prec-1.)/(nmock_prec-len(x)-2.)

    covar=covarin
    if np.size(np.shape(covarin)) == 1:
        err=covarin
        covar=np.zeros((np.size(err),np.size(err)))
        covar[np.arange(np.size(err)),np.arange(np.size(err))]=err**2
    
    #First do a SVD decomposition
    u,s,v=np.linalg.svd(covar)

    # start with minuit
    m,pf,err,cov,a,b=do_minuit(x,y,covarin,guess,functname) # a=chi2Minuit, b=dofMinuit
    ndim=np.size(guess)
    nwalkers=ndim*2
    nok=0
    print('\nStart emcee with '+np.str(ndim)+' dimensions and '+np.str(nwalkers)+' walkers')
    # initial guess
    p0=emcee.utils.sample_ball(np.array(pf),np.array(err)*3,nwalkers)
    # Initialize emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobcov, args=[x,y,s,v,functname],threads=nthreads)
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
    sampler.run_mcmc(pos, nbmc)
    # find chain for best walker
    chains=sampler.chain
    fractions=sampler.acceptance_fraction
    frac_threshold=0.5
    print('     Best fraction: '+np.str(max(fractions)))
    wfrac=where(fractions >= frac_threshold)
    print('     '+np.str(np.size(wfrac))+' walkers are above f='+np.str(frac_threshold))
    if max(fractions) > frac_threshold:
        best=np.where(fractions == max(fractions))
        thechain=chains[wfrac[0],:,:]
        sp=np.shape(thechain)
        thechain=np.reshape(thechain,[sp[0]*sp[1],sp[2]])
        
    parfit=np.zeros(ndim)
    for i in arange(ndim): parfit[i]=np.mean(thechain[:,i])
    errfit=np.zeros(ndim)
    for i in arange(ndim): errfit[i]=np.std(thechain[:,i])
    covariance=np.zeros((ndim,ndim))
    for i in arange(ndim):
        for j in arange(ndim):
            covariance[i,j]=np.mean((thechain[:,i]-parfit[i])*(thechain[:,j]-parfit[j]))

    chi2_out = -2.*lnprobcov(parfit, x, y, s, v, functname)

    return(thechain,parfit,errfit,covariance,chi2_out,np.size(x)-ndim)
        
################################################################








