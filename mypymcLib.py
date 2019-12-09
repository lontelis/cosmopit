import numpy as np
from numpy import *
import pymc
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from scipy.ndimage import gaussian_filter1d
import scipy
from scipy import integrate
from scipy import interpolate
from scipy import ndimage
#import numMath,cosmology
# Taken from J.C.Hamilton and remodified by P.Ntelis October 2016

print('This mypymclib new')
###############################################################################
########################## Monte-Carlo Markov-Chains Functions ################
###############################################################################
### define data classes #######################################################
class Data():
    def __init__(self, xvals=None, yvals=None, errors=None,  model=None, if_cross_Likelihood=False ,yvals_right=None, model_right=None, prior=False, nmock_prec=None):
        self.prior = prior
        self.model = model
        self.xvals = xvals
        self.yvals = yvals
        self.if_cross_Likelihood=if_cross_Likelihood
        if if_cross_Likelihood:
            self.model_right = model_right
            self.yvals_right = yvals_right
        if not self.prior:
            if np.size(np.shape(errors)) == 1:
                self.covar=np.zeros((np.size(errors),np.size(errors)))
                self.covar[np.arange(np.size(errors)),np.arange(np.size(errors))]=errors**2
            else:
                self.covar = errors
            if nmock_prec!=None: self.covar = self.covar * (nmock_prec-1.)/(nmock_prec-len(self.xvals)-2.)
            self.invcov = np.linalg.inv(self.covar)
    
    def __call__(self,*pars):
        if  not self.prior:
            if self.if_cross_Likelihood:
                val      =self.model(self.xvals,pars[0])
                val_right=self.model_right(self.xvals,pars[0])
                chi2     =np.dot(np.dot(self.yvals-val,self.invcov),self.yvals_right-val_right)
            else:
                val =self.model(self.xvals,pars[0])
                chi2=np.dot(np.dot(self.yvals-val,self.invcov),self.yvals-val)
        else:
            chi2 = self.model(self.xvals, pars[0])
        return(-0.5*chi2)


fid_test_linear = {
               'a': 1.0,
               'b': 0.0,
                }

def ll_test_linear(datasets, variables = ['a', 'b'], fidvalues = fid_test_linear):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    a      = pymc.Uniform('a', -4.,4.,   value = fid_test_linear['a'], observed = 'a' not in variables)
    b      = pymc.Uniform('b', -10.0,40.0, value = fid_test_linear['b'], observed = 'b' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, a=a,b=b):
        ll=0.
        pars = np.array([a,b])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

fid_test_linear_fixed_b = {
               'a': 2.0,
                }
def ll_test_linear_fixed_b(datasets, variables = ['a'], fidvalues = fid_test_linear_fixed_b):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    a      = pymc.Uniform('a', 0.0,4.,   value = fid_test_linear_fixed_b['a'], observed = 'a' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, a=a):
        ll=0.
        pars = np.array([a])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

fid_BR_nOmOX = {
               'n':  0.12,
               'Om': 0.31,
               'OX': 0.1,
                }

def ll_BR_nOmOX(datasets, variables = ['n', 'Om', 'OX'], fidvalues = fid_BR_nOmOX):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    n_min,n_max   = -1.5, 0.5 #-1.0, 0.5 
    Om_min,Om_max = 0.0, 1.0 #+0.0, 1.0 
    OX_min,OX_max = 0.0, 1.0 #+0.0, 1.0 

    n      = pymc.Uniform('n' , n_min,n_max  , value = fid_BR_nOmOX['n'] , observed = 'n'  not in variables)
    Om     = pymc.Uniform('Om', Om_min,Om_max, value = fid_BR_nOmOX['Om'], observed = 'Om' not in variables)
    OX     = pymc.Uniform('OX', OX_min,OX_max, value = fid_BR_nOmOX['OX'], observed = 'OX' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, n=n,Om=Om,OX=OX):
        ll=0.
        pars = np.array([n,Om,OX])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

fid_test_linear = {
               'a': 1.0,
               'b': 0.0,
                }

def ll_test_linear(datasets, variables = ['a', 'b'], fidvalues = fid_test_linear):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    a      = pymc.Uniform('a', -4.,4.,   value = fid_test_linear['a'], observed = 'a' not in variables)
    b      = pymc.Uniform('b', -10.0,40.0, value = fid_test_linear['b'], observed = 'b' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, a=a,b=b):
        ll=0.
        pars = np.array([a,b])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

fid_test_linear_fixed_b = {
               'a': 2.0,
                }
def ll_test_linear_fixed_b(datasets, variables = ['a'], fidvalues = fid_test_linear_fixed_b):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    a      = pymc.Uniform('a', 0.0,4.,   value = fid_test_linear_fixed_b['a'], observed = 'a' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, a=a):
        ll=0.
        pars = np.array([a])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_b0OmfNL = {
               'b0':5.0,
               #'h':0.67,
               'om0':0.3,
               'fNL':0.0,
                }

def Sll_model_b0OmfNL(datasets, variables = ['b0','om0','fNL'], fidvalues = Sfid_params_b0OmfNL):

    if (isinstance(datasets, list) is False): datasets=[datasets]
    b0     = pymc.Uniform('b0',    1.0,8.0 , value = Sfid_params_b0OmfNL['b0'] , observed = 'b0'  not in variables)
    om0     = pymc.Uniform('om0',  0.1,1.0 , value = Sfid_params_b0OmfNL['om0'] , observed = 'om0' not in variables) 
    fNL    = pymc.Uniform('fNL', -100.,100., value = Sfid_params_b0OmfNL['fNL'], observed = 'fNL' not in variables) 
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, b0=b0,om0=om0,fNL=fNL): 
        ll=0.
        pars = np.array([b0,om0,fNL]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_b0fNL = {
               'b0':1.0,
               #'h':0.67,
               'fNL':0.0,
                }

def Sll_model_b0fNL(datasets, variables = ['b0','fNL'], fidvalues = Sfid_params_b0fNL):

    if (isinstance(datasets, list) is False): datasets=[datasets]
    b0     = pymc.Uniform('b0',    0.0,5.0 , value = Sfid_params_b0fNL['b0'] , observed = 'b0'  not in variables)
    fNL    = pymc.Uniform('fNL', -300.,300., value = Sfid_params_b0fNL['fNL'], observed = 'fNL' not in variables) 
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, b0=b0,fNL=fNL): 
        ll=0.
        pars = np.array([b0,fNL]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_dcabrsrVomol = {
              'dc':-1.0,
              'a':3.5  ,
              'b':2.7  ,
              'rs':30  ,
              'rV':30  ,
              'om':0.32,
              'ol':1-0.32,
                }

def Sll_model_dcabrsrVomol(datasets, variables = ['dc','a','b','rs','rV','om','ol'], fidvalues = Sfid_params_dcabrsrVomol):
    if (isinstance(datasets, list) is False): datasets=[datasets]
        
    dc = pymc.Uniform('dc', -3.,0.0 , value = Sfid_params_dcabrsrVomol['dc'], observed = 'dc' not in variables)
    a  = pymc.Uniform('a' , 2.0,5.0 , value = Sfid_params_dcabrsrVomol['a'] , observed = 'a' not in variables)
    b  = pymc.Uniform('b' , 1.0,4.0, value = Sfid_params_dcabrsrVomol['b'] , observed = 'b' not in variables)
    rs = pymc.Uniform('rs', 10.,100., value = Sfid_params_dcabrsrVomol['rs'], observed = 'rs' not in variables)
    rV = pymc.Uniform('rV', 10.,100., value = Sfid_params_dcabrsrVomol['rV'], observed = 'rV' not in variables)
    om = pymc.Uniform('om', 0.0,2.0 , value = Sfid_params_dcabrsrVomol['om'], observed = 'om' not in variables)
    ol = pymc.Uniform('ol', 0.0,2.0 , value = Sfid_params_dcabrsrVomol['ol'], observed = 'ol' not in variables)

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0,dc=dc,a=a,b=b,rs=rs,rV=rV,om=om,ol=ol ):
        ll=0.
        pars = np.array([dc,a,b,rs,rV,om,ol])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_dcabrsrVom = {
              'dc':-1.0,
              'a':2.3  ,
              'b':7.0  ,
              'rs':30  ,
              'rV':30  ,
              'om':0.32,
                }

def Sll_model_dcabrsrVom(datasets, variables = ['dc','a','b','rs','rV','om'], fidvalues = Sfid_params_dcabrsrVom):
    if (isinstance(datasets, list) is False): datasets=[datasets]
        
    dc = pymc.Uniform('dc', -3.,0.0 , value = Sfid_params_dcabrsrVom['dc'], observed = 'dc' not in variables)
    a  = pymc.Uniform('a' , 0.0,3.0 , value = Sfid_params_dcabrsrVom['a'] , observed = 'a' not in variables)
    b  = pymc.Uniform('b' , 6.0,10.0, value = Sfid_params_dcabrsrVom['b'] , observed = 'b' not in variables)
    rs = pymc.Uniform('rs', 10.,100., value = Sfid_params_dcabrsrVom['rs'], observed = 'rs' not in variables)
    rV = pymc.Uniform('rV', 10.,100., value = Sfid_params_dcabrsrVom['rV'], observed = 'rV' not in variables)
    om = pymc.Uniform('om', 0.0,2.0 , value = Sfid_params_dcabrsrVom['om'], observed = 'om' not in variables)

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0,dc=dc,a=a,b=b,rs=rs,rV=rV,om=om ):
        ll=0.
        pars = np.array([dc,a,b,rs,rV,om])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_Aomol = {
               'A': 1.0, # 1.2
               #'B': 0.57, # 0.55
               'om': 0.31, #0.31,
               'ol': 0.69, #0.69,
                }

def Sll_model_Aomol(datasets, variables = ['A','om','ol'], fidvalues = Sfid_params_Aomol):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    A_min,A_max   = 0.1,2.00 #RH 1.0,2.50, 1.0,2.50 #0.5,4.0  # optimum! #2.15,2.25 still shit # 1.0,3.0
    om_min,om_max = 0.1,1.5 #0.2,0.60 #RH 0.2,0.60 #0.2,0.90 #0.0,1.0 but for interpolated Rh we need 0.2,1.0
    #ol_min,ol_max = 0.1,1.1 #0.55,0.75 #RH 0.5,0.80 #0.4,0.90 #0.0,1.0 but for interpolated Rh we need 0.4,1.0
    ol_min,ol_max = 0.1,1.5 #0.55,0.75 #RH 0.5,0.80 #0.4,0.90 #0.0,1.0 but for interpolated Rh we need 0.4,1.0
    A     = pymc.Uniform('A', A_min,A_max, value = Sfid_params_Aomol['A'], observed = 'A' not in variables)
    om     = pymc.Uniform('om', om_min,om_max, value = Sfid_params_Aomol['om'], observed = 'om' not in variables)
    ol     = pymc.Uniform('ol', ol_min,ol_max, value = Sfid_params_Aomol['ol'], observed = 'ol' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, A=A,om=om,ol=ol):
        ll=0.
        pars = np.array([A,om,ol])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_baiso = {
               'bias':2.0,
                'a':1.0
                }

def Sll_model_baiso(datasets, variables = ['bias','a'], fidvalues = Sfid_params_baiso):
    if (isinstance(datasets, list) is False): datasets=[datasets]

    bias     = pymc.Uniform('bias', 1.5,2.5, value = Sfid_params_baiso['bias'], observed = 'bias' not in variables)
    a        = pymc.Uniform('a', 0.7,1.3 ,   value = Sfid_params_baiso['a'],  observed = 'a'    not in variables)

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, bias=bias,a=a):
        ll=0.
        pars = np.array([bias,a])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_OmOLAsigmaA0A1A2 = {

               'Om':0.31,
               'OL':0.69,
               'A':0.001,
               'sigma':10,
               'A0':-0.1,
               'A1':-0.2,
               'A2':-100.,
                }

def Sll_model_OmOLAsigmaA0A1A2(datasets, variables = ['Om','OL','A','sigma','A0','A1','A2'], fidvalues = Sfid_params_OmOLAsigmaA0A1A2):

    if (isinstance(datasets, list) is False): datasets=[datasets]

    Om   = pymc.Uniform('Om', 0.0,1.0 ,     value = Sfid_params_OmOLAsigmaA0A1A2['Om'],   observed = 'Om'    not in variables)
    OL   = pymc.Uniform('OL', 0.0,1.0,   value = Sfid_params_OmOLAsigmaA0A1A2['OL'],observed = 'OL' not in variables) # 0.090,0.300                                                     
    A    = pymc.Uniform('A', 0.0,0.2, value = Sfid_params_OmOLAsigmaA0A1A2['A'], observed = 'A' not in variables)
    sigma= pymc.Uniform('sigma', 1.0,20.0, value = Sfid_params_OmOLAsigmaA0A1A2['sigma'], observed = 'sigma' not in variables)
    A0   = pymc.Uniform('A0', -1.0,1.0, value = Sfid_params_OmOLAsigmaA0A1A2['A0'], observed = 'A0' not in variables)    
    A1   = pymc.Uniform('A1', -20.0,20.0, value = Sfid_params_OmOLAsigmaA0A1A2['A1'], observed = 'A1' not in variables)    
    A2   = pymc.Uniform('A2', -100.0,100.0, value = Sfid_params_OmOLAsigmaA0A1A2['A2'], observed = 'A2' not in variables)    

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, Om=Om,OL=OL,A=A,sigma=sigma,A0=A0,A1=A1,A2=A2): 
        ll=0.
        pars = np.array([Om,OL,A,sigma,A0,A1,A2]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_OmOL = {

               'Om':0.31,
               'OL':0.69,

                }

def Sll_model_OmOL(datasets, variables = ['Om','OL'], fidvalues = Sfid_params_OmOL):

    if (isinstance(datasets, list) is False): datasets=[datasets]

    Om   = pymc.Uniform('Om', -1.0,1.0,   value = Sfid_params_OmOL['Om'],observed = 'Om'    not in variables)
    OL   = pymc.Uniform('OL', -1.0,1.0,   value = Sfid_params_OmOL['OL'],observed = 'OL' not in variables) # 0.090,0.300                                                     

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, Om=Om,OL=OL):
        ll=0.
        pars = np.array([Om,OL]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_w0OL = {

               'w0':-1.0,
               'OL':0.69,

                }

def Sll_model_w0OL(datasets, variables = ['w0','OL'], fidvalues = Sfid_params_w0OL):

    if (isinstance(datasets, list) is False): datasets=[datasets]

    w0   = pymc.Uniform('w0', -2.0,0.0 ,  value = Sfid_params_w0OL['w0'],observed = 'w0' not in variables)
    OL   = pymc.Uniform('OL', -1.0,1.0,   value = Sfid_params_w0OL['OL'],observed = 'OL' not in variables) # 0.090,0.300                                                     

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, w0=w0,OL=OL):
        ll=0.
        pars = np.array([w0,OL]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_A = {
               'A': 1.45, #1.5, #2.2,
               #'om': 0.31,
               #'ol': 0.69,
                }

def Sll_model_A(datasets, variables = ['A'], fidvalues = Sfid_params_A):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    
    #A_min,A_max=1.5,2.8#2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0
    #B_min,B_max=-2,1.  #0.545,0.555     # optimum!   

    #A_min,A_max=1.6,2.4 #1.0,2.0#2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0
    #B_min,B_max=0.0,2.0 #0.545,0.555     # optimum!   

    A_min,A_max=0.5,2.0 
    A     = pymc.Uniform('A', A_min,A_max, value = Sfid_params_A['A'], observed = 'A' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, A=A):
        ll=0.
        pars = np.array([A,])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_AB = {
               'A': 2.0, #1.5, #2.2,
               'B': 0.55,
               #'om': 0.31,
               #'ol': 0.69,
                }

def Sll_model_AB(datasets, variables = ['A','B'], fidvalues = Sfid_params_AB):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    
    #A_min,A_max=1.5,2.8#2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0
    #B_min,B_max=-2,1.  #0.545,0.555     # optimum!   

    #A_min,A_max=1.6,2.4 #1.0,2.0#2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0
    #B_min,B_max=0.0,2.0 #0.545,0.555     # optimum!   

    A_min,A_max=0.0,3.0 
    B_min,B_max=-1,3.0 

    A     = pymc.Uniform('A', A_min,A_max, value = Sfid_params_AB['A'], observed = 'A' not in variables)
    B     = pymc.Uniform('B', B_min,B_max, value = Sfid_params_AB['B'], observed = 'B' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, A=A,B=B):
        ll=0.
        pars = np.array([A,B,])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

#### END: MODELS and Bounds #####

def run_mcmc(data,niter=80000, nburn=20000, nthin=1, variables=['Om', 'Ol', 'w'], external=None, w_ll_model='LCDMsimple',delay=1000,if_get_chains_chi2=True):
    if w_ll_model=='LCDM':
        feed_ll_model= ll_model
        feedPars     = fid_params
    elif w_ll_model=='LCDMsimple': # simple LCDM
        feed_ll_model= Sll_model
        feedPars     = Sfid_params
    elif w_ll_model=='LCDMsimple_ebSimplify':
        feed_ll_model= Sll_model_ebSimplify
        feedPars     = Sfid_params_ebSimplify
    elif w_ll_model=='LCDMsimple_ebSimplify_h':
        feed_ll_model= Sll_model_ebSimplify_h
        feedPars     = Sfid_params_ebSimplify_h
    elif w_ll_model=='LCDMsimple_5bhocdm':
        feed_ll_model= Sll_model_5bhocdm
        feedPars     = Sfid_params_5bhocdm
    elif w_ll_model=='LCDMsimple_bhocdm':
        feed_ll_model= Sll_model_bhocdm
        feedPars     = Sfid_params_bhocdm
    elif w_ll_model=='LCDMsimple_5bocdm':
        feed_ll_model= Sll_model_5bocdm
        feedPars     = Sfid_params_5bocdm
    elif w_ll_model=='LCDMsimple_bocdm':
        feed_ll_model= Sll_model_bocdm
        feedPars     = Sfid_params_bocdm
    elif w_ll_model=='LCDMsimple_bocdmw0': #b ocdm w0
        feed_ll_model= Sll_model_bocdmw0
        feedPars     = Sfid_params_eb
    elif w_ll_model=='LCDMsimple_bocdmOk': #b ocdm Ok
        feed_ll_model= Sll_model_bocdmOk
        feedPars     = Sfid_params_eb
    elif w_ll_model=='LCDMsimple_5bocdmw0': #5b ocdm w0                                                                                                                                         
        feed_ll_model= Sll_model_5bocdmw0
        feedPars     = Sfid_params_eb
    elif w_ll_model=='LCDMsimple_5bocdmOk': #5b ocdm Ok                                                                                                                                         
        feed_ll_model= Sll_model_5bocdmOk
        feedPars     = Sfid_params_eb
    elif w_ll_model=='LCDMsimple_eb': # simple LCDM 5b h  ocdm                  
        feed_ll_model= Sll_model_eb
        feedPars     = Sfid_params_eb
    elif w_ll_model=='LCDMsimple_ebk': # simple LCDM 5b h ocdm Ok                          
        feed_ll_model= Sll_model_ebk
        feedPars     = Sfid_params_eb
    elif w_ll_model=='LCDMsimple_ebw': # simple LCDM 5b                           
        feed_ll_model= Sll_model_ebw
        feedPars     = Sfid_params_eb
    elif w_ll_model=='sBAO':
        feed_ll_model = ll_modelBAO
        feedPars      = fidParsBAO 
    elif w_ll_model=='sBAOLCDM':
        feed_ll_model = ll_modelBAOLCDM
        feedPars      = fidParsBAOLCDM  
    elif w_ll_model=='lowxi_rh':
        print('lowxi_rh')
        feed_ll_model = ll_model_lowxi_rh
        feedPars      = fidPars_lowxi_rh
    elif w_ll_model=='LCDMsimple_5bomolhrd':
        feed_ll_model = Sll_model_5bomol_hrd
        feedPars      = Sfid_params_5bomol_hrd
    elif w_ll_model=='LCDMsimple_5bomokhrd':
        feed_ll_model = Sll_model_5bomok_hrd
        feedPars      = Sfid_params_5bomok_hrd
    elif w_ll_model=='LCDMsimple_5b5aiso':
        feed_ll_model = Sll_model_5b5aiso
        feedPars      = Sfid_params_5b5aiso
    elif w_ll_model=='LCDMsimple_5broadband5Gaussian':
        feed_ll_model = Sll_model_5broadband5Gaussian
        feedPars      = Sfid_params_5broadband5Gaussian
    elif w_ll_model=='LCDMsimple_broadbandGaussian':
        feed_ll_model = Sll_model_broadbandGaussian
        feedPars      = Sfid_params_broadbandGaussian
    elif w_ll_model=='LCDMsimple_5bomol':
        feed_ll_model = Sll_model_5bomol
        feedPars      = Sfid_params_5bomol
    elif w_ll_model=='LCDMsimple_omolhrd':
        feed_ll_model = Sll_model_omolhrd
        feedPars      = Sfid_params_omolhrd
    elif w_ll_model=='LCDMsimple_omolhrh':
        feed_ll_model = Sll_model_omolhrh
        feedPars      = Sfid_params_omolhrh
    elif w_ll_model=='LCDMsimple_omolh':
        feed_ll_model = Sll_model_omolh
        feedPars      = Sfid_params_omolh
    elif w_ll_model=='bias_aiso':
        feed_ll_model = Sll_model_baiso
        feedPars      = Sfid_params_baiso
    elif w_ll_model=='LCDMsimple_omol_hrd_AB':
        feed_ll_model = Sll_model_omol_hrd_AB
        feedPars      = Sfid_params_omol_hrd_AB
    elif w_ll_model=='LCDMsimple_ABomolhrh':
        feed_ll_model = Sll_model_ABomolhrh
        feedPars      = Sfid_params_ABomolhrh
    elif w_ll_model=='LCDMsimple_ABomol':
        feed_ll_model = Sll_model_ABomol
        feedPars      = Sfid_params_ABomol
    elif w_ll_model=='LCDMsimple_Aomol':
        feed_ll_model = Sll_model_Aomol
        feedPars      = Sfid_params_Aomol
    elif w_ll_model=='LCDMsimple_b0omgammamg':
        feed_ll_model = Sll_model_b0omgammamg
        feedPars      = Sfid_params_b0omgammamg
    elif w_ll_model=='LCDMsimple_AB':
        feed_ll_model = Sll_model_AB
        feedPars      = Sfid_params_AB
    elif w_ll_model=='LCDMsimple_A':
        feed_ll_model = Sll_model_A
        feedPars      = Sfid_params_A
    elif w_ll_model=='LCDMsimple_omolAB':
        feed_ll_model = Sll_model_omolAB
        feedPars      = Sfid_params_omolAB
    elif w_ll_model=='LCDMsimple_Aom':
        feed_ll_model = Sll_model_Aom
        feedPars      = Sfid_params_Aom
    elif w_ll_model=='LCDMsimple_okom':
        feed_ll_model = Sll_model_okom
        feedPars      = Sfid_params_okom
    elif w_ll_model=='LCDMsimple_omolb0b1b2':
        feed_ll_model = Sll_model_omolb0b1b2
        feedPars      = Sfid_params_omolb0b1b2
    elif w_ll_model =='LCDMsimple_omolwb0':
        feed_ll_model = Sll_model_omolwb0
        feedPars      = Sfid_params_omolwb0
    elif w_ll_model=='BR_model':
        feed_ll_model = ll_BR
        feedPars      = fid_BR
    elif w_ll_model=='BR_nOm':
        feed_ll_model = ll_BR_nOm
        feedPars      = fid_BR_nOm
    elif w_ll_model=='BR_nOmOX':
        feed_ll_model = ll_BR_nOmOX
        feedPars      = fid_BR_nOmOX
    elif w_ll_model=='test_linear':
      feed_ll_model  = ll_test_linear
      feedPars       = fid_test_linear      
    elif w_ll_model=='test_linear_fixed_b':
      feed_ll_model  = ll_test_linear_fixed_b
      feedPars       = fid_test_linear_fixed_b 

    elif w_ll_model=='LCDM_b0OmfNL':
      feed_ll_model  = Sll_model_b0OmfNL
      feedPars       = Sfid_params_b0OmfNL
    elif w_ll_model=='LCDM_b0fNL':
      feed_ll_model  = Sll_model_b0fNL
      feedPars       = Sfid_params_b0fNL
    elif w_ll_model=='dcabrsrVom':
      feed_ll_model = Sll_model_dcabrsrVom
      feedPars       = Sfid_params_dcabrsrVom
    elif w_ll_model=='dcabrsrVomol':
      feed_ll_model = Sll_model_dcabrsrVomol
      feedPars       = Sfid_params_dcabrsrVomol
    elif w_ll_model=='OmOLAsigmaA0A1A2':
      feed_ll_model = Sll_model_OmOLAsigmaA0A1A2
      feedPars       = Sfid_params_OmOLAsigmaA0A1A2
    elif w_ll_model=='OmOL':
      feed_ll_model = Sll_model_OmOL
      feedPars       = Sfid_params_OmOL
    elif w_ll_model=='w0OL':
      feed_ll_model = Sll_model_w0OL
      feedPars       = Sfid_params_w0OL

    chain = pymc.MCMC(feed_ll_model(data, variables, fidvalues=feedPars))
    chain.use_step_method(pymc.AdaptiveMetropolis,chain.stochastics,delay=delay)
    chain.sample(iter=niter,burn=nburn,thin=nthin)
    ch ={}
    for v in variables: ch[v] = chain.trace(v)[:]
    if if_get_chains_chi2:
        chi2 = -2.*chain.logp
        print('chains,chi2')
        return ch,chi2
    else: 
        return ch

def burnChains(chains,kmin=0):
    newChains=dict(chains) # dict(chains)
    # python 2:  kmax = newChains[newChains.keys()[0]].size
    # python 3:  kmax = np.size(newChains[next(iter(newChains))])
    kmax = np.size(newChains[next(iter(newChains))])
    for k in newChains.keys(): newChains[k] = newChains[k][kmin:kmax]
    return newChains

#### PLOTTING
def matrixplot(chain,vars,col,sm,limits=None,nbins=None,doit=None,alpha=0.7,labels=None,Blabel=None,Blabelsize=20,plotCorrCoef=True,plotScatter=False,NsigLim=3,ChangeLabel=False,Bpercentile=False,kk=0,plotNumberContours='12',paper2=True,plotLegendLikelihood=True,if_condition_chain=False):
    '''
    kk=5 # kk==0 gives all parameters
           kk!=0 removes kk first parameters
    plotNumberContours= '1', '2', '12'
                        '1'  1sigma contour 
                        '2'  2sigma contours 
                        '12' 1sigma and 2sigma contours
    '''
    nplots=len(vars)
    if labels is None: labels = vars
    if doit is None: doit=np.repeat([True],nplots)
    mm=np.zeros(nplots)
    ss=np.zeros(nplots)
    for i in np.arange(nplots):
        if vars[i] in chain.keys():
            mm[i]=np.mean( chain[vars[i]] )
            ss[i]=np.std(  chain[vars[i]] )
    if limits is None:
        limits=[]
        for i in np.arange(nplots):
            limits.append([mm[i]-NsigLim*ss[i],mm[i]+NsigLim*ss[i]]) # 3
    if if_condition_chain:
        condition_chain=np.where((chain['A']>1.3)&(chain['A']<1.4))
        for var in chain.keys():
            chain[var] = chain[var][condition_chain]

    num=0
    for i in np.arange(nplots-kk)+kk:
         for j in np.arange(nplots-kk)+kk:
            num+=1
            if (i == j):
                a=subplot(nplots-kk,nplots-kk,num)
                a.tick_params(labelsize=8)
                if i == 0: ylabel('$\mathcal{L}($'+labels[i]+'$)/\mathcal{L}_{max}$',size=20)
                if i == nplots-1: xlabel(labels[j],size=20)
                var=vars[j]
                if vars[i]=='bias': xlim( [mm[i]-20*ss[i],mm[i]+20*ss[i]] )
                else: xlim( limits[i] )
                
                ylim(0,1.2)
                if (var in chain.keys()) and (doit[j]==True):
                    if nbins is None: nbins=100 
                    bla=np.histogram(chain[var],bins=nbins,normed=True)
                    xhist=(bla[1][0:nbins]+bla[1][1:nbins+1])/2
                    yhist=gaussian_filter1d(bla[0],ss[i]/5/(xhist[1]-xhist[0]),mode='constant',order=0,truncate=3)
                    mode_xhist=xhist[ np.argmax(yhist) ]
                    if Bpercentile:
                        mm = np.mean(chain[var])
                        p25= np.percentile(chain[var],100-68) - mm 
                        p75= np.percentile(chain[var],68)     - mm
                        plot(xhist,yhist/max(yhist),color=col,label='%0.2f $\pm_{%0.2f}^{+%0.2f}$, mode=%0.2f'%(mm, p25,p75, mode_xhist ))
                    else:
                        plot(xhist,yhist/max(yhist),color=col,label='%0.3f $\pm$ %0.3f, mode=%0.2f'%(np.mean(chain[var]), np.std(chain[var]) , mode_xhist ) )
                    if paper2=='2018':
                      ylim([0.,1.1])

                      if vars[j]=='om': xlim([0.0,1.0])
                    if paper2=='2019':
                      ylim([0.,1.0])

                      if vars[j]=='om': xlim([0.2,0.7]) #xlim([0.2,0.6]) #0.0,0.6
                      if vars[j]=='ol': xlim([0.3,1.0]) #xlim([0.5,0.8]) #0.4,1.0
                      if vars[j]=='A' : xlim([1.0,1.8])
                    if paper2=='2020':
                      ylim([0.,2.0])
                      if vars[j]=='b0': xlim(0.86,0.90) #xlim(0.75,1.01)
                      if vars[j]=='fNL': xlim(-100,100) #xlim(0.75,1.01)
                    else:
                      ylim([0.,3.0])
                    if plotLegendLikelihood: legend(frameon=False,fontsize=8) # 12##8 12 15

            if (i>j):
                a=subplot(nplots-kk,nplots-kk,num)
                a.tick_params(labelsize=12) #8 15

                var0=labels[j]
                var1=labels[i]
                if paper2=='2018' and vars[j]=='om':
                  print(vars[j])
                  xlim([0.0,1.0])
                elif paper2=='2019':
                  if   vars[j]=='om': xlim([0.2,0.7]) #xlim([0.2,0.6]) #0.0,0.6
                  elif vars[j]=='ol': xlim([0.3,1.0]) #xlim([0.5,0.8]) #0.4,1.0
                  elif vars[j]=='A':  xlim([1.0,1.8]) #1.2,2.5
                  if   vars[i]=='om': ylim([0.2,0.7]) # ylim([0.1,0.8])#ylim([0.2,0.6]) #0.0,0.6
                  elif vars[i]=='ol': ylim([0.3,1.0]) # ylim([0.2,1.0]) #ylim([0.5,0.8]) #0.4,1.0
                  elif vars[i]=='A':  ylim([1.0,1.8]) # ylim([0.9,1.9]) #1.2,2.5
                elif paper2=='2020':
                  ylim(limits[i])  
                  xlim(limits[j])
                  ylim(-100.,100.)#                # ylim(-100.,100.)
                  xlim(0.86,0.90) #xlim(0.75,1.01) # xlim(0.86,1.01)
                  

                else:
                  if vars[j]=='bias': xlim( [mm[j]-20*ss[j],mm[j]+20*ss[j]] )
                  else:               xlim(limits[j])

                  ylim(limits[i])
                if i == nplots-1: xlabel(var0,size=20)
                if j == 0+kk: ylabel(var1,size=20)    
                if (vars[i] in chain.keys()) and (vars[j] in chain.keys()) and (doit[j]==True) and (doit[i]==True):
                    if plotNumberContours=='12':
                        levels = [0.9545,0.6827]
                    elif plotNumberContours=='1':
                        levels = [0.6827]
                    elif plotNumberContours=='2':
                        levels = [0.9545]
                    a0=cont(chain[vars[j]],chain[vars[i]],levels=levels,color=col,nsmooth=sm,alpha=alpha,plotCorrCoef=plotCorrCoef,plotScatter=plotScatter) 
                
    if Blabel:
        frame1=plt.subplot(nplots,nplots,nplots) #,facecolor='white')
        frame1.plot(1,1,col,label=Blabel)
        frame1.legend(loc=1,numpoints=1,frameon=False,prop={'size':Blabelsize}) #25 15
        frame1.set_frame_on(False)
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        frame1.axes.get_xaxis().set_ticks([]) 
        frame1.axes.get_yaxis().set_ticks([])        

    subplots_adjust(wspace=0.3,hspace=0.3)
    #subplots_adjust(wspace=0.15,hspace=0.1)
    #subplots_adjust(wspace=0.,hspace=0.)
    return(a0)

def getcols(color):
    color=str(color)
    if (color == 'blue') or (color == 'b'):
        cols=['SkyBlue','MediumBlue']
    elif (color == 'red') or (color == 'r'):
        cols=['LightCoral','Red']
    elif (color == 'green') or color == 'g':
        cols=['LightGreen','Green']
    elif (color == 'pink') or (color == 'm'):
        cols=['LightPink','HotPink']
    elif (color == 'yellow') or (color == 'y'):
        cols=['Yellow','Gold']
    elif (color == 'black') or (color == 'k'):
        cols=['grey','black']
    elif color == 'orange':
        cols=['Coral','OrangeRed']
    elif color == 'purple':
        cols=['Violet','DarkViolet']
    elif color == 'brown':
        cols=['BurlyWood','SaddleBrown']
    return(cols)

def get_array_colors():
    return np.array(['blue','red','green','pink','yellow','black','orange','purple','brown'])

def cont(x,y,xlim=None,ylim=None,levels=[0.9545,0.6827],alpha=0.7,color='blue',
     nbins=256,
     nsmooth=4,Fill=True,plotCorrCoef=True,plotScatter=False,**kwargs):

    levels.sort()
    levels.reverse()
    cols=getcols(color)
    dx=np.max(x)-np.min(x)
    dy=np.max(y)-np.min(y)
    if xlim is None: xlim=[np.min(x)-dx/3,np.max(x)+dx/3]
    if ylim is None: ylim=[np.min(y)-dy/3,np.max(y)+dy/3]
    range=[xlim,ylim]

    a,xmap,ymap=scipy.histogram2d(x,y,bins=256,range=range)
    a=np.transpose(a)
    xmap=xmap[:-1]
    ymap=ymap[:-1]
    dx=xmap[1]-xmap[0]
    dy=ymap[1]-ymap[0]
    z=scipy.ndimage.filters.gaussian_filter(a,nsmooth)
    z=z/np.sum(z)/dx/dy
    sz=np.sort(z.flatten())[::-1]
    cumsz=integrate.cumtrapz(sz)
    cumsz=cumsz/max(cumsz)
    f=interpolate.interp1d(cumsz,np.arange(np.size(cumsz)))
    indices=f(levels).astype('int')
    vals=sz[indices].tolist()
    vals.append(np.max(sz))
    vals.sort()
    
    if Fill:
        for i in np.arange(np.size(levels)):
            contourf(xmap, ymap, z, vals[i:i+2],colors=cols[i],alpha=alpha,**kwargs)
    else:
        contour(xmap, ymap, z, vals[0:1],colors=cols[1],**kwargs)
        contour(xmap, ymap, z, vals[1:2],colors=cols[1],**kwargs)

    a=Rectangle((np.max(xmap),np.max(ymap)),0.1,0.1,fc=cols[1])
    
    if plotScatter: scatter(x,y,color=color,marker=u'.')
    if plotCorrCoef:
        mmx,ssx = x.mean(),x.std()
        mmy,ssy = y.mean(),y.std()
        rcorrcoeff = np.corrcoef(x,y)[0,1]
        #if abs(rcorrcoeff)>0.65:
        label_cont='$\\rho$=%0.2f'%(rcorrcoeff)
        xarr = np.array([mmx-ssx,mmx+ssx])
        yarr = np.array([mmy-ssy,mmy+ssy])
        plot(xarr,xarr*0.0+mmy,color,label=label_cont)
        plot(xarr*0.0+mmx,yarr,color) 
        legend(loc=2,frameon=False,numpoints=1,fontsize=12) #8 15 20

    return(a)

###############################################################################
###############################################################################

######### Template Running the pymc #######                                                                                                                                            
def run_pymc(xvals,yvals,covin,model,nmock_prec=1000, addPrior=False,gauss_prior_cmb=None,vars=[],niter=1e4,nburn=0,w_ll_model='lowxi_rh'):
    data = Data(xvals=xvals,yvals=yvals,errors=covin,model=model,nmock_prec=nmock_prec)

    if addPrior:
        prior_cmb = Data(model=gauss_prior_cmb, prior=True)
        chains    = run_mcmc([prior_cmb,data], variables=vars,niter=niter,nburn=nburn,w_ll_model=w_ll_model)
    else:
        chains    = run_mcmc(data, variables=vars,niter=niter,nburn=nburn,w_ll_model=w_ll_model)

    nchains   = numMath.burnChains(chains,kmin=niter/2)
    FitedPars = np.array([nchains[k] for k in sort(nchains.keys())]).mean(axis=1)
    chi2      = -2.*data(FitedPars)
    y_model   = model(xvals,FitedPars)
    return chains,nchains,FitedPars,chi2,y_model


'''
# OLD MCMC scenarios to search if you miss someone! :)
fid_params = {'h':0.67 , 'ob':0.022,'ocdm':0.118 ,'ns':0.96, 'ln1e10As':3.094 , 'Omega_Lambda':0.658 ,'Omega_k':0.0004 , 'w0':-1.004, 'wa':1e-4,'bias':1.95, 'sigp':350.}
def ll_model(datasets, variables = ['h', 'ns', 'ln1e10As', 'ob', 'ocdm', 'Omega_k' ,'w0','wa'], fidvalues = fid_params):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    h        = pymc.Uniform('h', 0.60,0.80 ,     value = fid_params['h']   ,  observed = 'h'    not in variables)
    ob       = pymc.Uniform('ob',   0.0150,0.0350, value = fid_params['ob'],  observed = 'ob'   not in variables)
    ocdm     = pymc.Uniform('ocdm', 0.100,0.150,   value = fid_params['ocdm'],observed = 'ocdm' not in variables)
    ns       = pymc.Uniform('ns', 0.900,1.100 ,    value = fid_params['ns'],  observed = 'ns'   not in variables)
    ln1e10As = pymc.Uniform('ln1e10As', 2.80,3.20 ,  value = fid_params['ln1e10As'], observed = 'ln1e10As' not in variables)
    Omega_k  = pymc.Uniform('Omega_k', -0.05,0.05, value = fid_params['Omega_k'], observed = 'Omega_k' not in variables)
    Omega_L  = pymc.Uniform('Omega_Lambda', -0.5,0.7, value = fid_params['Omega_Lambda'], observed = 'Omega_Lambda' not in variables)
    w0       = pymc.Uniform('w0', -2,-0.05, value = fid_params['w0'], observed = 'w0' not in variables)
    wa       = pymc.Uniform('wa', -0.1,0.1, value = fid_params['wa'], observed = 'w0' not in variables)
    ###
    bias     = pymc.Uniform('bias', 1.5,2.5, value = fid_params['bias'], observed = 'bias' not in variables)
    sigp     = pymc.Uniform('sigp', 250,450, value = fid_params['sigp'], observed = 'sigp' not in variables)   
    @pymc.stochastic(trace=True,observed=True,plot=False)
    #def loglikelihood(value=0, Om=Om,Ol=Ol,h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,w=w):
    def loglikelihood(value=0, h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,Omega_L=Omega_L,Omega_k=Omega_k,w0=w0,wa=wa,bias=bias,sigp=sigp):
        ll=0.
        pars = np.array([h,ob,ocdm,ns,ln1e10As,Omega_L,Omega_k,w0,wa,bias,sigp])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    
    return(locals())


Sfid_params = {
               'bias':1.95,
               #'alpha':1.0
               'h':0.67, 
               'ocdm':0.118,
               #'ns':0.96, 
               #'ln1e10As':3.094, 
               'Omega_k':0.0040, 
               #'Omega_Lambda':0.658,
               #'w0':-1.004, 
               #'wa':1e-4 
            }


def Sll_model(datasets, variables = ['bias','ocdm','Omega_k'], fidvalues = Sfid_params): # 'ocdm','Omega_k','w0','wa', 'alpha'
    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias     = pymc.Uniform('bias', 1.6,2.4, value = Sfid_params['bias'], observed = 'bias' not in variables)
    #alpha    = pymc.Uniform('alpha', 0.8,1.2, value = Sfid_params['alpha'], observed = 'alpha' not in variables)
    h        = pymc.Uniform('h', 0.60,0.80 ,     value = Sfid_params['h'],   observed = 'h'    not in variables)
    #ob       = pymc.Uniform('ob',   0.0150,0.0350, value = Sfid_params['ob'],  observed = 'ob'   not in variables)
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params['ocdm'],observed = 'ocdm' not in variables)
    #ns       = pymc.Uniform('ns', 0.900,1.100 ,    value = Sfid_params['ns'],  observed = 'ns'   not in variables)
    #ln1e10As = pymc.Uniform('ln1e10As', 2.80,3.20 ,  value = Sfid_params['ln1e10As'], observed = 'ln1e10As' not in variables)
    Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params['Omega_k'], observed = 'Omega_k' not in variables)
    #Omega_L  = pymc.Uniform('Omega_Lambda', -0.5,0.7, value = Sfid_params['Omega_Lambda'], observed = 'Omega_Lambda' not in variables)
    #w0       = pymc.Uniform('w0', -2.5,-0.05, value = Sfid_params['w0'], observed = 'w0' not in variables)
    #wa       = pymc.Uniform('wa', -0.1,0.1, value = Sfid_params['wa'], observed = 'w0' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    #def loglikelihood(value=0, Om=Om,Ol=Ol,h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,w=w):
    def loglikelihood(value=0, bias=bias,h=h,ocdm=ocdm,Omega_k=Omega_k): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocdm,Omega_k=Omega_k,w0=w0,wa=wa,
        ll=0.
        pars = np.array([bias,h,ocdm,Omega_k]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_eb = {
               'bias0':1.88,'bias1':1.85,'bias2':1.95,'bias3':2.00,'bias4':2.15,
               #'alpha':1.0                                                                                                                                                            
               'h':0.67,                                                                                                                                                              
               'ocdm':0.118,
               #'ns':0.96,                                                                                                                                                             
               #'ln1e10As':3.094,                                                                                                                                                      
               'Omega_k':-0.040,
               #'Omega_Lambda':0.658,                                                                                                                                                  
               'w0_fld':-1.004,                                                                                                                                                           
               #'wa':1e-4                                                                                                                                                              
                }

def Sll_model_eb(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','h','ocdm','Omega_k','w0_fld'], fidvalues = Sfid_params_eb): # 'ocdm','Omega_k','w0','wa', 'alpha'                                                             
    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_eb['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_eb['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_eb['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_eb['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,2.4, value = Sfid_params_eb['bias4'], observed = 'bias4' not in variables)
    #alpha    = pymc.Uniform('alpha', 0.8,1.2, value = Sfid_params['alpha'], observed = 'alpha' not in variables)                                                            
    h        = pymc.Uniform('h', 0.50,0.90 ,     value = Sfid_params_eb['h'],   observed = 'h'    not in variables)
    #ob       = pymc.Uniform('ob',   0.0150,0.0350, value = Sfid_params['ob'],  observed = 'ob'   not in variables)                                            
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_eb['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300
    #ns       = pymc.Uniform('ns', 0.900,1.100 ,    value = Sfid_params['ns'],  observed = 'ns'   not in variables)                                                 
    #ln1e10As = pymc.Uniform('ln1e10As', 2.80,3.20 ,  value = Sfid_params['ln1e10As'], observed = 'ln1e10As' not in variables)                                        
    Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params_eb['Omega_k'], observed = 'Omega_k' not in variables)
    #Omega_L  = pymc.Uniform('Omega_Lambda', -0.5,0.7, value = Sfid_params['Omega_Lambda'], observed = 'Omega_Lambda' not in variables)                                                                    
    w0_fld       = pymc.Uniform('w0_fld', -3.0,-0.01, value = Sfid_params_eb['w0_fld'], observed = 'w0_fld' not in variables)                                                                      
    #wa       = pymc.Uniform('wa', -0.1,0.1, value = Sfid_params['wa'], observed = 'w0' not in variables)                                                                                  
    @pymc.stochastic(trace=True,observed=True,plot=False)
    #def loglikelihood(value=0, Om=Om,Ol=Ol,h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,w=w):                                                                                                                                                     
    def loglikelihood(value=0, bias0=bias0, bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,h=h,ocdm=ocdm,Omega_k=Omega_k,w0_fld=w0_fld): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocdm,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,                                                                                                                                                       
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,h,ocdm,Omega_k,w0_fld]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,
                                                                                        
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_ebSimplify = {
               'bias0':1.88,'bias1':1.85,'bias2':1.95,'bias3':2.00,'bias4':2.15,
               #'alpha':1.0                                                                                                                                                           \
                                                                                                                                                                                       
               #'h':0.67,
               'ocdm':0.118,
               #'ns':0.96,                                                                                                                                                            \
                                                                                                                                                                                       
               #'ln1e10As':3.094,                                                                                                                                                     \
                                                                                                                                                                                       
               'Omega_k':0.0,
               'Omega_Lambda':0.658,                                                                                                                                                 \
                                                                                                                                                                                       
#           'w0_fld':-1.004,                                                                                                                                                       \

           #'wa':1e-4                                                                                                                                                             \
                                                                                                                                                                                       
                }


def Sll_model_ebSimplify(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','ocdm','Omega_Lambda'], fidvalues = Sfid_params_ebSimplify): # 'ocdm','Omega_k','w0','wa', 'alpha'    \
                                                                                                                                                                                       
    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_ebSimplify['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_ebSimplify['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_ebSimplify['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_ebSimplify['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,2.4, value = Sfid_params_ebSimplify['bias4'], observed = 'bias4' not in variables)
    #alpha    = pymc.Uniform('alpha', 0.8,1.2, value = Sfid_params['alpha'], observed = 'alpha' not in variables)                                                                      
    #h        = pymc.Uniform('h', 0.50,0.90 ,     value = Sfid_params_eb['h'],   observed = 'h'    not in variables)
    #ob       = pymc.Uniform('ob',   0.0150,0.0350, value = Sfid_params['ob'],  observed = 'ob'   not in variables)                                                                    
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_ebSimplify['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                    
    #ns       = pymc.Uniform('ns', 0.900,1.100 ,    value = Sfid_params['ns'],  observed = 'ns'   not in variables)                                                                    
    #ln1e10As = pymc.Uniform('ln1e10As', 2.80,3.20 ,  value = Sfid_params['ln1e10As'], observed = 'ln1e10As' not in variables)                                                         
    #Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params_eb['Omega_k'], observed = 'Omega_k' not in variables)
    Omega_Lambda  = pymc.Uniform('Omega_Lambda', 0.0,1.0, value = Sfid_params_ebSimplify['Omega_Lambda'], observed = 'Omega_Lambda' not in variables)                                               
                                                                                                                                                                                       
    #w0_fld       = pymc.Uniform('w0_fld', -3.0,-0.01, value = Sfid_params_eb['w0_fld'], observed = 'w0_fld' not in variables)                                                         \

    #wa       = pymc.Uniform('wa', -0.1,0.1, value = Sfid_params['wa'], observed = 'w0' not in variables)                                                                             \
                                                                                                                                                                                       
    @pymc.stochastic(trace=True,observed=True,plot=False)
    #def loglikelihood(value=0, Om=Om,Ol=Ol,h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,w=w):        
    def loglikelihood(value=0, bias0=bias0, bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,ocdm=ocdm,Omega_Lambda=Omega_Lambda): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocd\m,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,                                                                                                                                                                                                                                                                                                                                      
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,ocdm,Omega_Lambda]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                              

        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_ebSimplify_h = {
               'bias0':1.88,'bias1':1.85,'bias2':1.95,'bias3':2.00,'bias4':2.15,
               #'alpha':1.0                                                                                                                                                           \                          

               'h':0.67,                                                                                                                                                                                        
               'ocdm':0.118,
               #'ns':0.96,                                                                                                                                                            \                          

               #'ln1e10As':3.094,                                                                                                                                                     \                          

           #'Omega_k':0.0040,                                                                                                                                                                                
               'Omega_Lambda':0.658,                                                                                                                                                 \

#           'w0_fld':-1.004,                                                                                                                                                       \                             

           #'wa':1e-4                                                                                                                                                             \                              

                }

def Sll_model_ebSimplify_h(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','h','ocdm','Omega_Lambda'], fidvalues = Sfid_params_ebSimplify): # 'ocdm','Omega_k','w0','wa', 'alpha'    \                  

    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_ebSimplify['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_ebSimplify['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_ebSimplify['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_ebSimplify['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,2.4, value = Sfid_params_ebSimplify['bias4'], observed = 'bias4' not in variables)
    #alpha    = pymc.Uniform('alpha', 0.8,1.2, value = Sfid_params['alpha'], observed = 'alpha' not in variables)                                                                                                
    h        = pymc.Uniform('h', 0.50,0.90 ,     value = Sfid_params_ebSimplify['h'],   observed = 'h'    not in variables)                                                                                             
    #ob       = pymc.Uniform('ob',   0.0150,0.0350, value = Sfid_params['ob'],  observed = 'ob'   not in variables)                                                                                              
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_ebSimplify['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                                      
    #ns       = pymc.Uniform('ns', 0.900,1.100 ,    value = Sfid_params['ns'],  observed = 'ns'   not in variables)                                                                                              
    #ln1e10As = pymc.Uniform('ln1e10As', 2.80,3.20 ,  value = Sfid_params['ln1e10As'], observed = 'ln1e10As' not in variables)                                                                                   
    #Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params_eb['Omega_k'], observed = 'Omega_k' not in variables)                                                                                      
    Omega_Lambda  = pymc.Uniform('Omega_Lambda', 0.0,1.0, value = Sfid_params_ebSimplify['Omega_Lambda'], observed = 'Omega_Lambda' not in variables)

    #w0_fld       = pymc.Uniform('w0_fld', -3.0,-0.01, value = Sfid_params_eb['w0_fld'], observed = 'w0_fld' not in variables)                                                         \                         

    #wa       = pymc.Uniform('wa', -0.1,0.1, value = Sfid_params['wa'], observed = 'w0' not in variables)                                                                             \                          

    @pymc.stochastic(trace=True,observed=True,plot=False)
    #def loglikelihood(value=0, Om=Om,Ol=Ol,h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,w=w):                                                                                                                    
    def loglikelihood(value=0, bias0=bias0, bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,h=h,ocdm=ocdm,Omega_Lambda=Omega_Lambda): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocd\m,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,       
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,h,ocdm,Omega_Lambda]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                                                            

        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_5bhocdm = {
               'bias0':1.88,'bias1':1.85,'bias2':1.95,'bias3':2.00,'bias4':2.15,
               'h':0.67,                                                                                                                                                       
               'ocdm':0.118,
                }

def Sll_model_5bhocdm(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','h','ocdm'], fidvalues = Sfid_params_5bhocdm): # 'ocdm','Omega_k','w0','wa', 'alpha'    \                                                                                                                        
    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_5bhocdm['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_5bhocdm['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_5bhocdm['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_5bhocdm['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,2.4, value = Sfid_params_5bhocdm['bias4'], observed = 'bias4' not in variables)
    h        = pymc.Uniform('h', 0.50,0.90 ,     value = Sfid_params_5bhocdm['h'],   observed = 'h'    not in variables)
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_5bhocdm['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                                                                                                                                                  

    @pymc.stochastic(trace=True,observed=True,plot=False)
    #def loglikelihood(value=0, Om=Om,Ol=Ol,h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,w=w):                                                                                                                                                                                                                                
    def loglikelihood(value=0, bias0=bias0, bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,h=h,ocdm=ocdm): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocd\m,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,                                                                                                    
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,h,ocdm]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                                                                                                                                                                      

        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_5bocdm = {
               'bias0':1.88,'bias1':1.85,'bias2':1.95,'bias3':2.00,'bias4':2.15,
               #'h':0.67,
               'ocdm':0.118,
                }
def Sll_model_5bocdm(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','ocdm'], fidvalues = Sfid_params_5bhocdm): # 'ocdm','Omega_k','w0','wa', 'alpha'    \                  

    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_5bocdm['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_5bocdm['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_5bocdm['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_5bocdm['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,2.4, value = Sfid_params_5bocdm['bias4'], observed = 'bias4' not in variables)
    #h        = pymc.Uniform('h', 0.50,0.90 ,     value = Sfid_params_5bhocdm['h'],   observed = 'h'    not in variables)
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_5bocdm['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                        
    @pymc.stochastic(trace=True,observed=True,plot=False)
    
    def loglikelihood(value=0, bias0=bias0, bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,ocdm=ocdm): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocd\m,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,ocdm]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                                                      
        
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_bhocdm = {
               'bias':2.0,
               'h':0.67,
               'ocdm':0.118,
                }

def Sll_model_bhocdm(datasets, variables = ['bias','h','ocdm'], fidvalues = Sfid_params_bhocdm):

    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias     = pymc.Uniform('bias', 1.6,2.4, value = Sfid_params_bhocdm['bias'], observed = 'bias' not in variables)
    h        = pymc.Uniform('h', 0.50,0.90 ,     value = Sfid_params_bhocdm['h'],   observed = 'h'    not in variables)
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_bhocdm['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                             
    @pymc.stochastic(trace=True,observed=True,plot=False)

    def loglikelihood(value=0, bias=bias,h=h,ocdm=ocdm): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocd\m,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,      
        ll=0.
        pars = np.array([bias,h,ocdm]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                                                          

        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_bocdm = {
               'bias':2.0,
               #'h':0.67,
               'ocdm':0.118,
                }

def Sll_model_bocdm(datasets, variables = ['bias','ocdm'], fidvalues = Sfid_params_bhocdm):

    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias     = pymc.Uniform('bias', 1.6,2.4, value = Sfid_params_bhocdm['bias'], observed = 'bias' not in variables)
    #h        = pymc.Uniform('h', 0.50,0.90 ,     value = Sfid_params_bhocdm['h'],   observed = 'h'    not in variables)
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_bhocdm['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                         
        
    @pymc.stochastic(trace=True,observed=True,plot=False)

    def loglikelihood(value=0, bias=bias,ocdm=ocdm): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocd\m,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,                                                        
        ll=0.
        pars = np.array([bias,ocdm]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                                                                                

        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

def Sll_model_ebk(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','h','ocdm','Omega_k'], fidvalues = Sfid_params_eb): # 'ocdm','Omega_k','w0','wa', 'alpha'
    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_eb['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_eb['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_eb['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_eb['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,2.4, value = Sfid_params_eb['bias4'], observed = 'bias4' not in variables)
    #alpha    = pymc.Uniform('alpha', 0.8,1.2, value = Sfid_params['alpha'], observed = 'alpha' not in variables)                                                            
    h        = pymc.Uniform('h', 0.60,0.80 ,     value = Sfid_params_eb['h'],   observed = 'h'    not in variables)
    #ob       = pymc.Uniform('ob',   0.0150,0.0350, value = Sfid_params['ob'],  observed = 'ob'   not in variables)                                            
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_eb['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300
    #ns       = pymc.Uniform('ns', 0.900,1.100 ,    value = Sfid_params['ns'],  observed = 'ns'   not in variables)                                                 
    #ln1e10As = pymc.Uniform('ln1e10As', 2.80,3.20 ,  value = Sfid_params['ln1e10As'], observed = 'ln1e10As' not in variables)                                        
    Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params_eb['Omega_k'], observed = 'Omega_k' not in variables)
    #Omega_L  = pymc.Uniform('Omega_Lambda', -0.5,0.7, value = Sfid_params['Omega_Lambda'], observed = 'Omega_Lambda' not in variables)                                                         
    #w0_fld       = pymc.Uniform('w0_fld', -2.5,-0.05, value = Sfid_params_eb['w0_fld'], observed = 'w0_fld' not in variables)                                                                  
    #wa       = pymc.Uniform('wa', -0.1,0.1, value = Sfid_params['wa'], observed = 'w0' not in variables)                                                                                  
    @pymc.stochastic(trace=True,observed=True,plot=False)
    #def loglikelihood(value=0, Om=Om,Ol=Ol,h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,w=w):                                                                                                  
    def loglikelihood(value=0, bias0=bias0, bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,h=h,ocdm=ocdm,Omega_k=Omega_k): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocdm,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,                                                                                                                                                       
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,h,ocdm,Omega_k]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,
                                                                                        
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

def Sll_model_5bocdmOk(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','ocdm','Omega_k'], fidvalues = Sfid_params_eb): # 'ocdm','Omega_k','w0','wa', 'alpha'                     
    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_eb['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_eb['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_eb['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_eb['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,2.4, value = Sfid_params_eb['bias4'], observed = 'bias4' not in variables)
    #alpha    = pymc.Uniform('alpha', 0.8,1.2, value = Sfid_params['alpha'], observed = 'alpha' not in variables)                                                                                
    #h        = pymc.Uniform('h', 0.60,0.80 ,     value = Sfid_params_eb['h'],   observed = 'h'    not in variables)
    #ob       = pymc.Uniform('ob',   0.0150,0.0350, value = Sfid_params['ob'],  observed = 'ob'   not in variables)                                                                              
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_eb['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                              
    #ns       = pymc.Uniform('ns', 0.900,1.100 ,    value = Sfid_params['ns'],  observed = 'ns'   not in variables)                                                                              
    #ln1e10As = pymc.Uniform('ln1e10As', 2.80,3.20 ,  value = Sfid_params['ln1e10As'], observed = 'ln1e10As' not in variables)                                                                   
    Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params_eb['Omega_k'], observed = 'Omega_k' not in variables)
    #Omega_L  = pymc.Uniform('Omega_Lambda', -0.5,0.7, value = Sfid_params['Omega_Lambda'], observed = 'Omega_Lambda' not in variables)                                                         
    #w0_fld       = pymc.Uniform('w0_fld', -2.5,-0.05, value = Sfid_params_eb['w0_fld'], observed = 'w0_fld' not in variables)                                                                  
    #wa       = pymc.Uniform('wa', -0.1,0.1, value = Sfid_params['wa'], observed = 'w0' not in variables)                                                                                        
    @pymc.stochastic(trace=True,observed=True,plot=False)
    #def loglikelihood(value=0, Om=Om,Ol=Ol,h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,w=w):                                                                                                    
    def loglikelihood(value=0, bias0=bias0, bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,ocdm=ocdm,Omega_k=Omega_k): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocdm,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,                                                                                                                                                                                   
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,ocdm,Omega_k]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                                               

        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


def Sll_model_bocdmOk(datasets, variables = ['bias','ocdm','Omega_k'], fidvalues = Sfid_params_eb): # 'ocdm','Omega_k','w0','wa', 'alpha'                      

    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias     = pymc.Uniform('bias', 1.6,2.4, value = Sfid_params_eb['bias3'], observed = 'bias' not in variables)
    
    #alpha    = pymc.Uniform('alpha', 0.8,1.2, value = Sfid_params['alpha'], observed = 'alpha' not in variables)                                                                               
    #h        = pymc.Uniform('h', 0.60,0.80 ,     value = Sfid_params_eb['h'],   observed = 'h'    not in variables)
    #ob       = pymc.Uniform('ob',   0.0150,0.0350, value = Sfid_params['ob'],  observed = 'ob'   not in variables)                                                                             
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_eb['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                           
    #ns       = pymc.Uniform('ns', 0.900,1.100 ,    value = Sfid_params['ns'],  observed = 'ns'   not in variables)                                                                             
    #ln1e10As = pymc.Uniform('ln1e10As', 2.80,3.20 ,  value = Sfid_params['ln1e10As'], observed = 'ln1e10As' not in variables)                                                                  
    Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params_eb['Omega_k'], observed = 'Omega_k' not in variables)
    #Omega_L  = pymc.Uniform('Omega_Lambda', -0.5,0.7, value = Sfid_params['Omega_Lambda'], observed = 'Omega_Lambda' not in variables)                                                        \
    #w0_fld       = pymc.Uniform('w0_fld', -2.5,-0.05, value = Sfid_params_eb['w0_fld'], observed = 'w0_fld' not in variables)                                                                 
    #wa       = pymc.Uniform('wa', -0.1,0.1, value = Sfid_params['wa'], observed = 'w0' not in variables)                                                                                       
    @pymc.stochastic(trace=True,observed=True,plot=False)
    #def loglikelihood(value=0, Om=Om,Ol=Ol,h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,w=w):                                                                                                  \
        
    def loglikelihood(value=0, bias=bias ,ocdm=ocdm,Omega_k=Omega_k): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocdm,Omega_k=Omega_k,w0_fld\=w0_fld,wa=wa,                                                                                                                                                                                   
        ll=0.
        pars = np.array([bias,ocdm,Omega_k]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                                               

        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

def Sll_model_ebw(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','h','ocdm','w0_fld'], fidvalues = Sfid_params_eb): # 'ocdm','Omega_k','w0','wa', 'alpha'                                                              

    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_eb['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_eb['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_eb['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_eb['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,2.4, value = Sfid_params_eb['bias4'], observed = 'bias4' not in variables)
    #alpha    = pymc.Uniform('alpha', 0.8,1.2, value = Sfid_params['alpha'], observed = 'alpha' not in variables)                                                            
    h        = pymc.Uniform('h', 0.60,0.80 ,     value = Sfid_params_eb['h'],   observed = 'h'    not in variables)
    #ob       = pymc.Uniform('ob',   0.0150,0.0350, value = Sfid_params['ob'],  observed = 'ob'   not in variables)                                            
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_eb['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300
    #ns       = pymc.Uniform('ns', 0.900,1.100 ,    value = Sfid_params['ns'],  observed = 'ns'   not in variables)                                                 
    #ln1e10As = pymc.Uniform('ln1e10As', 2.80,3.20 ,  value = Sfid_params['ln1e10As'], observed = 'ln1e10As' not in variables)                                        
    #Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params_eb['Omega_k'], observed = 'Omega_k' not in variables)
    #Omega_L  = pymc.Uniform('Omega_Lambda', -0.5,0.7, value = Sfid_params['Omega_Lambda'], observed = 'Omega_Lambda' not in variables)                                                                    
    w0_fld       = pymc.Uniform('w0_fld', -2.5,-0.05, value = Sfid_params_eb['w0_fld'], observed = 'w0_fld' not in variables)                                                                      
    #wa       = pymc.Uniform('wa', -0.1,0.1, value = Sfid_params['wa'], observed = 'w0' not in variables)                                                                                  
    @pymc.stochastic(trace=True,observed=True,plot=False)
    #def loglikelihood(value=0, Om=Om,Ol=Ol,h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,w=w):                                                                                                                                                     
    def loglikelihood(value=0, bias0=bias0, bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,h=h,ocdm=ocdm,w0_fld=w0_fld): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocdm,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,                                                                                                                                                       
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,h,ocdm,w0_fld]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,
                                                                                        
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

def Sll_model_5bw0(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','ocdm','w0_fld'], fidvalues = Sfid_params_eb): # 'ocdm','Omega_k','w0','wa', 'alpha'                       
    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_eb['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_eb['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_eb['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_eb['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,2.4, value = Sfid_params_eb['bias4'], observed = 'bias4' not in variables)
    #alpha    = pymc.Uniform('alpha', 0.8,1.2, value = Sfid_params['alpha'], observed = 'alpha' not in variables)                                                                               
    #h        = pymc.Uniform('h', 0.60,0.80 ,     value = Sfid_params_eb['h'],   observed = 'h'    not in variables)
    #ob       = pymc.Uniform('ob',   0.0150,0.0350, value = Sfid_params['ob'],  observed = 'ob'   not in variables)                                                                             
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_eb['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                             
    #ns       = pymc.Uniform('ns', 0.900,1.100 ,    value = Sfid_params['ns'],  observed = 'ns'   not in variables)                                                                             
    #ln1e10As = pymc.Uniform('ln1e10As', 2.80,3.20 ,  value = Sfid_params['ln1e10As'], observed = 'ln1e10As' not in variables)                                                                  
    #Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params_eb['Omega_k'], observed = 'Omega_k' not in variables)                                                                     
    #Omega_L  = pymc.Uniform('Omega_Lambda', -0.5,0.7, value = Sfid_params['Omega_Lambda'], observed = 'Omega_Lambda' not in variables)                                                         
    w0_fld       = pymc.Uniform('w0_fld', -2.5,-0.05, value = Sfid_params_eb['w0_fld'], observed = 'w0_fld' not in variables)                                                                   
    #wa       = pymc.Uniform('wa', -0.1,0.1, value = Sfid_params['wa'], observed = 'w0' not in variables)                                                                                       
    @pymc.stochastic(trace=True,observed=True,plot=False)
    #def loglikelihood(value=0, Om=Om,Ol=Ol,h=h,ob=ob,ocdm=ocdm,ns=ns,ln1e10As=ln1e10As,w=w):                                                                                                  
    
    def loglikelihood(value=0, bias0=bias0, bias1=bias1, bias2=bias2, bias3=bias3, bias4=bias4,ocdm=ocdm,w0_fld=w0_fld): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocdm,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,                                                                                                                                                                                     
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,ocdm,w0_fld]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                                               
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

def Sll_model_bocdmw0(datasets, variables = ['bias','ocdm','w0_fld'], fidvalues = Sfid_params_eb): # 'ocdm','Omega_k','w0','wa', 'alpha'                       

    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias      = pymc.Uniform('bias', 1.6,2.4, value = Sfid_params_eb['bias3'], observed = 'bias' not in variables)
    #alpha    = pymc.Uniform('alpha', 0.8,1.2, value = Sfid_params['alpha'], observed = 'alpha' not in variables)                                                                                
    #h        = pymc.Uniform('h', 0.60,0.80 ,     value = Sfid_params_eb['h'],   observed = 'h'    not in variables)
    #ob       = pymc.Uniform('ob',   0.0150,0.0350, value = Sfid_params['ob'],  observed = 'ob'   not in variables)                                                                              
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_eb['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                            
    #ns       = pymc.Uniform('ns', 0.900,1.100 ,    value = Sfid_params['ns'],  observed = 'ns'   not in variables)                                                                             
    #ln1e10As = pymc.Uniform('ln1e10As', 2.80,3.20 ,  value = Sfid_params['ln1e10As'], observed = 'ln1e10As' not in variables)                                                                  
    #Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params_eb['Omega_k'], observed = 'Omega_k' not in variables)                                                                     
    #Omega_L  = pymc.Uniform('Omega_Lambda', -0.5,0.7, value = Sfid_params['Omega_Lambda'], observed = 'Omega_Lambda' not in variables)                                                        
    w0_fld       = pymc.Uniform('w0_fld', -2.5,-0.05, value = Sfid_params_eb['w0_fld'], observed = 'w0_fld' not in variables)                                                         

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, bias=bias,ocdm=ocdm,w0_fld=w0_fld): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocdm,Omega_k=Omega_k,w0_fld=w\0_fld,wa=wa,                                                                                                                                                                                     
        ll=0.
        pars = np.array([bias,ocdm,w0_fld]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                                                

        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_bhocdmk = {
               'bias':2.0,
               'h':0.67,
               'ocdm':0.118,
               'Omega_k':-0.040,
                }

def Sll_model_bhocdmk(datasets, variables = ['bias','h','ocdm','Omega_k'], fidvalues = Sfid_params_bhocdmk):

    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias     = pymc.Uniform('bias', 1.6,2.4, value = Sfid_params_bhocdmk['bias'], observed = 'bias' not in variables)
    h        = pymc.Uniform('h', 0.50,0.90 ,     value = Sfid_params_bhocdmk['h'],   observed = 'h'    not in variables)
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_bhocdmk['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                     
    Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params_bhocdmk['Omega_k'], observed = 'Omega_k' not in variables)

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, bias=bias,h=h,ocdm=ocdm,Omega_k=Omega_k): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocd\m,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,                                     
        ll=0.
        pars = np.array([bias,h,ocdm,Omega_k]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                                                                             
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_bhocdmw = {
               'bias':2.0,
               'h':0.67,
               'ocdm':0.118,
               'w0_fld':-1.004,
                }

def Sll_model_bhocdmw(datasets, variables = ['bias','h','ocdm','w0_fld'], fidvalues = Sfid_params_bhocdmk):

    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias     = pymc.Uniform('bias', 1.6,2.4, value = Sfid_params_bhocdmw['bias'], observed = 'bias' not in variables)
    h        = pymc.Uniform('h', 0.50,0.90 ,     value = Sfid_params_bhocdmw['h'],   observed = 'h'    not in variables)
    ocdm     = pymc.Uniform('ocdm', 0.090,0.300,   value = Sfid_params_bhocdmw['ocdm'],observed = 'ocdm' not in variables) # 0.090,0.300                                                     
    w0       = pymc.Uniform('w0_fld', -2.5,-0.05,   value = Sfid_params_bhocdmw['w0_fld'],observed = 'w0_fld' not in variables) # 0.090,0.300
    #Omega_k  = pymc.Uniform('Omega_k', -0.5,0.5, value = Sfid_params_bhocdmk['Omega_k'], observed = 'Omega_k' not in variables)

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, bias=bias,h=h,ocdm=ocdm,w0=w0): #bias=bias,alpha=alpha,h=h,ob=ob,ocdm=ocd\m,Omega_k=Omega_k,w0_fld=w0_fld,wa=wa,                                     
        ll=0.
        pars = np.array([bias,h,ocdm,w0]) #bias,alpha,ocdm,Omega_k,w0,wa, #ob,Omega_L,Omega_k,w0,wa,                                                                                    

        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

fidParsBAO = {'p0':104. ,'p1':0.008,'p2':-2.5,'p3':150.,'p4':0.005, 'p5':10.0,}
def ll_modelBAO(datasets, variables=['p0','p1','p2','p3','p4','p5'], fidvalues=fidParsBAO):
    """ 
      xi(r,p) = p4*exp( -0.5(r-p0)**2/(p5**2) ) +  bb(p1,p2,p3) 
      where bb(p1,p2,p3) = p1 + p2/r + p3/r**2
    """
    if (isinstance(datasets, list) is False): datasets=[datasets]
    #p0 = pymc.Uniform('p0',90.,130.   ,fidvalues['p0'], observed='p0' not in variables )
    #p1 = pymc.Uniform('p1',2.,8.      ,fidvalues['p1'], observed='p1' not in variables )
    #p2 = pymc.Uniform('p2',-0.02,0.02 ,fidvalues['p2'], observed='p2' not in variables )
    #p3 = pymc.Uniform('p3',-10.,5.0   ,fidvalues['p3'], observed='p3' not in variables )
    #p4 = pymc.Uniform('p4',10.0,200.  ,fidvalues['p4'], observed='p4' not in variables )
    #p5 = pymc.Uniform('p5',-0.02,0.02 ,fidvalues['p5'], observed='p5' not in variables )#
    # 0->0 1->5 2->1 3->2 4->3 5->1

    p0 = pymc.Uniform('p0',80.,140.   ,fidvalues['p0'], observed='p0' not in variables )
    p1 = pymc.Uniform('p1',-0.02,0.02 ,fidvalues['p1'], observed='p1' not in variables )
    p2 = pymc.Uniform('p2',-5.,5.     ,fidvalues['p2'], observed='p2' not in variables )
    p3 = pymc.Uniform('p3',50.0,300.  ,fidvalues['p3'], observed='p3' not in variables )
    p4 = pymc.Uniform('p4',-0.02,0.02 ,fidvalues['p4'], observed='p4' not in variables )
    p5 = pymc.Uniform('p5',0.,20.     ,fidvalues['p5'], observed='p5' not in variables )    

    @pymc.stochastic(trace=True,observed=True,plot=False)

    def loglikelihood(value=0, p0=p0,p1=p1,p2=p2,p3=p3,p4=p4,p5=p5):
        ll=0.
        pars = np.array([p0,p1,p2,p3,p4,p5])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

fidParsBAOLCDM = {'p0':1.04 ,
                  'p1':0.08,
                  'p2':-.1,
                  'p3':0.1,
                  'p4':2.0, 
                  #'p5':300.0
                  }

def ll_modelBAOLCDM(datasets, variables=['p0','p1','p2','p3','p4'], fidvalues=fidParsBAOLCDM):
    """ xi(r,p) = xi_LCDM_FID( p0*x, b=p4,sigp=p5, galaxy=True,kaiser=True,damping=True) +  bb(p1,p2,p3) """
    if (isinstance(datasets, list) is False): datasets=[datasets]
    #p0 = pymc.Uniform('p0',90.,130.   ,fidvalues['p0'], observed='p0' not in variables )
    #p1 = pymc.Uniform('p1',2.,8.      ,fidvalues['p1'], observed='p1' not in variables )
    #p2 = pymc.Uniform('p2',-0.02,0.02 ,fidvalues['p2'], observed='p2' not in variables )
    #p3 = pymc.Uniform('p3',-10.,5.0   ,fidvalues['p3'], observed='p3' not in variables )
    #p4 = pymc.Uniform('p4',10.0,200.  ,fidvalues['p4'], observed='p4' not in variables )
    #p5 = pymc.Uniform('p5',-0.02,0.02 ,fidvalues['p5'], observed='p5' not in variables )#
    # 0->0 1->5 2->1 3->2 4->3 5->1
    
    p0 = pymc.Uniform('p0',0.6,1.40   ,fidvalues['p0'], observed='p0' not in variables )
    p1 = pymc.Uniform('p1',-10.,10. ,fidvalues['p1'], observed='p1' not in variables )
    p2 = pymc.Uniform('p2',-10.,10.     ,fidvalues['p2'], observed='p2' not in variables )
    p3 = pymc.Uniform('p3',-200.,200.  ,fidvalues['p3'], observed='p3' not in variables )
    p4 = pymc.Uniform('p4',1.8,2.5    ,fidvalues['p4'], observed='p4' not in variables ) 
    #p5 = pymc.Uniform('p5',200.,500.     ,fidvalues['p5'], observed='p5' not in variables )    #!# SOS

    @pymc.stochastic(trace=True,observed=True,plot=False)

    def loglikelihood(value=0, p0=p0,p1=p1,p2=p2,p3=p3,p4=p4): # ,p5=p5
        ll=0.
        pars = np.array([p0,p4]) # p1,p2,p3, ,p5
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

fidPars_lowxi_rh = {#'p0':1.04 ,
                    #'p1':0.08,
                    #'p2':-.1,
                    #'p3':0.1,
                    'b':2.0,
                    #'p5':320,
                    'h':0.7,                    
                    'Om':0.3,
                    'Ol':0.7,
                    }

def ll_model_lowxi_rh(datasets, variables=['p0','p1','p2','p3','p4','Om','Ol'], fidvalues=fidPars_lowxi_rh):
    """ 
        Model
        xi(r,p) = xi_LCDM_FID( p0*x, b=p4, galaxy=True) +  bb(p1,p2,p3) 
        rh(b,OFid)*a_iso(Otrue) 
    """
    if (isinstance(datasets, list) is False): datasets=[datasets]
    print 'IMPLEMENT ME'
    
    #p1 = pymc.Uniform('p1',-10.,10. ,fidvalues['p1'], observed='p1' not in variables )
    #p2 = pymc.Uniform('p2',-10.,10.     ,fidvalues['p2'], observed='p2' not in variables )
    #p3 = pymc.Uniform('p3',-200.,200.  ,fidvalues['p3'], observed='p3' not in variables )
    b = pymc.Uniform('b',1.5,2.5       ,fidvalues['b'], observed='b' not in variables ) 
    #p5 = pymc.Uniform('p5',200.,500.     ,fidvalues['p5'], observed='p5' not in variables )
    h = pymc.Uniform('h',0.0,1.5       ,fidvalues['h'], observed='h' not in variables ) 
    Om = pymc.Uniform('Om',0.0,1.2     ,fidvalues['Om'], observed='Om' not in variables ) 
    Ol = pymc.Uniform('Ol',0.0,1.2     ,fidvalues['Ol'], observed='Ol' not in variables )

    @pymc.stochastic(trace=True,observed=True,plot=False)

    def loglikelihood(value=0, b=b,h=h,Om=Om,Ol=Ol): #p0=p0,p1=p1,p2=p2,p3=p3, p5=p5
        ll=0.
        pars = np.array([b,h,Om,Ol]) #p0,p1,p2,p3 ,p5
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_5bomol_hrd = {
               'bias0':1.88,'bias1':1.85,'bias2':1.95,'bias3':2.00,'bias4':2.15,
                'om':0.3156,                                                      
                'ol':1.-0.3156,
                'hrd':0.6727*149.2
                }

def Sll_model_5bomol_hrd(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','om','ol','hrd'], fidvalues = Sfid_params_5bomol_hrd):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_5bomol_hrd['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_5bomol_hrd['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_5bomol_hrd['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_5bomol_hrd['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,3.0, value = Sfid_params_5bomol_hrd['bias4'], observed = 'bias4' not in variables)
    om        = pymc.Uniform('om', -1.0,2.0 ,     value = Sfid_params_5bomol_hrd['om'],   observed = 'om'    not in variables)
    ol        = pymc.Uniform('ol', -1.0,2.0,   value = Sfid_params_5bomol_hrd['ol'],observed = 'ol' not in variables) # 0.090,0.300                                                                
    hrd       = pymc.Uniform('hrd', 40.0,160.0,   value = Sfid_params_5bomol_hrd['hrd'],observed = 'hrd' not in variables) # 0.090,0.300                                                           
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, bias0=bias0,bias1=bias1,bias2=bias2,bias3=bias3,bias4=bias4,om=om,ol=ol,hrd=hrd):
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,om,ol,hrd]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_5bomol = {
               'bias0':1.88,'bias1':1.85,'bias2':1.95,'bias3':2.00,'bias4':2.15,
                'om':0.3156,                                                      
                'ol':1.-0.3156,
                }
def Sll_model_5bomol(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','om','ol'], fidvalues = Sfid_params_5bomol):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_5bomol['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_5bomol['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_5bomol['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_5bomol['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,3.0, value = Sfid_params_5bomol['bias4'], observed = 'bias4' not in variables)
    om        = pymc.Uniform('om', -1.0,2.0 ,  value = Sfid_params_5bomol['om'],   observed = 'om'     not in variables)
    ol        = pymc.Uniform('ol', -1.0,2.0,   value = Sfid_params_5bomol['ol'],   observed = 'ol'     not in variables) # 0.090,0.300                                                                
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, bias0=bias0,bias1=bias1,bias2=bias2,bias3=bias3,bias4=bias4,om=om,ol=ol):
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,om,ol]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_5bomok_hrd = {
               'bias0':1.88,'bias1':1.85,'bias2':1.95,'bias3':2.00,'bias4':2.15,
                'om':0.3156,
                'ok':0.00,
                'hrd':0.6727*149.2
                }
def Sll_model_5bomok_hrd(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','om','ok','hrd'], fidvalues = Sfid_params_5bomok_hrd):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_5bomok_hrd['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_5bomok_hrd['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_5bomok_hrd['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_5bomok_hrd['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.6,3.0, value = Sfid_params_5bomok_hrd['bias4'], observed = 'bias4' not in variables)
    om        = pymc.Uniform('om', -1.0,2.0 ,     value = Sfid_params_5bomok_hrd['om'],   observed = 'om'    not in variables)
    ok        = pymc.Uniform('ok', -0.1,0.1,   value = Sfid_params_5bomok_hrd['ok'],observed = 'ok' not in variables) # 0.090,0.300                                                                
    hrd       = pymc.Uniform('hrd', 40.0,160.0,   value = Sfid_params_5bomok_hrd['hrd'],observed = 'hrd' not in variables) # 0.090,0.300                                                           
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, bias0=bias0,bias1=bias1,bias2=bias2,bias3=bias3,bias4=bias4,om=om,ok=ok,hrd=hrd):
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,om,ok,hrd])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_5b5aiso = {
               'bias0':1.88,'bias1':1.85,'bias2':1.95,'bias3':2.00,'bias4':2.15,
                'a0':1.0,'a1':1.0,'a2':1.0,'a3':1.0,'a4':1.0,
                }

def Sll_model_5b5aiso(datasets, variables = ['bias0','bias1','bias2','bias3','bias4','a0','a1','a2','a3','a4'], fidvalues = Sfid_params_5b5aiso):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    bias0     = pymc.Uniform('bias0', 1.6,2.4, value = Sfid_params_5b5aiso['bias0'], observed = 'bias0' not in variables)
    bias1     = pymc.Uniform('bias1', 1.6,2.4, value = Sfid_params_5b5aiso['bias1'], observed = 'bias1' not in variables)
    bias2     = pymc.Uniform('bias2', 1.6,2.4, value = Sfid_params_5b5aiso['bias2'], observed = 'bias2' not in variables)
    bias3     = pymc.Uniform('bias3', 1.6,2.4, value = Sfid_params_5b5aiso['bias3'], observed = 'bias3' not in variables)
    bias4     = pymc.Uniform('bias4', 1.7,3.0, value = Sfid_params_5b5aiso['bias4'], observed = 'bias4' not in variables)
    a0        = pymc.Uniform('a0', 0.7,1.3 ,   value = Sfid_params_5b5aiso['a0'],   observed = 'a0'    not in variables)
    a1        = pymc.Uniform('a1', 0.7,1.3 ,   value = Sfid_params_5b5aiso['a1'],   observed = 'a1'    not in variables)
    a2        = pymc.Uniform('a2', 0.7,1.3 ,   value = Sfid_params_5b5aiso['a2'],   observed = 'a2'    not in variables)
    a3        = pymc.Uniform('a3', 0.7,1.3 ,   value = Sfid_params_5b5aiso['a3'],   observed = 'a3'    not in variables)
    a4        = pymc.Uniform('a4', 0.1,1.1 ,   value = Sfid_params_5b5aiso['a4'],   observed = 'a4'    not in variables)

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, bias0=bias0,bias1=bias1,bias2=bias2,bias3=bias3,bias4=bias4,a0=a0,a1=a1,a2=a2,a3=a3,a4=a4):
        ll=0.
        pars = np.array([bias0,bias1,bias2,bias3,bias4,a0,a1,a2,a3,a4])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_5broadband5Gaussian = {
               'b0':0.005,
               'b1':0.005,
               'b2':0.005,
               'b3':0.005,
               'b4':0.005,
                'a0':1.0,
                'a1':1.0,
                'a2':1.0,
                'a3':1.0,
                'a4':1.0,
                'sigma0':10.0,
                'sigma1':10.0,
                'sigma2':10.0,
                'sigma3':10.0,
                'sigma4':10.0,
                'Alpha00':0.008,
                'Alpha01':0.008,
                'Alpha02':0.008,
                'Alpha03':0.008,
                'Alpha04':0.008,
                'Alpha10':-2.5,
                'Alpha11':-2.5,
                'Alpha12':-2.5,
                'Alpha13':-2.5,
                'Alpha14':-2.5,
                'Alpha20':150.0,
                'Alpha21':150.0,
                'Alpha22':150.0,
                'Alpha23':150.0,
                'Alpha24':150.0,
                }

def Sll_model_5broadband5Gaussian(datasets, variables = ['b0','b1','b2','b3','b4','a0','a1','a2','a3','a4',
          'sigma0',
          'sigma1',
          'sigma2',
          'sigma3',
          'sigma4',
          'Alpha00',
          'Alpha01',
          'Alpha02',
          'Alpha03',
          'Alpha04',
          'Alpha10',
          'Alpha11',
          'Alpha12',
          'Alpha13',
          'Alpha14',
          'Alpha20',
          'Alpha21',
          'Alpha22',
          'Alpha23',
          'Alpha24',]
          , fidvalues = Sfid_params_5broadband5Gaussian):
    """ 
      xi(r;z) = Sum_z bz* exp(-0.5*(r - az*get_rs() )**2/(sigmaz)**2 ) + Alpha0z + Alpha1z/r + Alpha2z/r**2 
      conected with rpymc_xi_ALLz_aiso.py
      5 redshift bins
    """
    if (isinstance(datasets, list) is False): datasets=[datasets]
    b0     = pymc.Uniform('b0', 0.000,0.01, value = Sfid_params_5broadband5Gaussian['b0'], observed = 'b0' not in variables)
    b1     = pymc.Uniform('b1', 0.000,0.01, value = Sfid_params_5broadband5Gaussian['b1'], observed = 'b1' not in variables)
    b2     = pymc.Uniform('b2', 0.000,0.01, value = Sfid_params_5broadband5Gaussian['b2'], observed = 'b2' not in variables)
    b3     = pymc.Uniform('b3', 0.000,0.01, value = Sfid_params_5broadband5Gaussian['b3'], observed = 'b3' not in variables)
    b4     = pymc.Uniform('b4', 0.000,0.01, value = Sfid_params_5broadband5Gaussian['b4'], observed = 'b4' not in variables)
    a0     = pymc.Uniform('a0', 0.8,1.2 ,   value = Sfid_params_5broadband5Gaussian['a0'],   observed = 'a0'    not in variables)
    a1     = pymc.Uniform('a1', 0.8,1.2 ,   value = Sfid_params_5broadband5Gaussian['a1'],   observed = 'a1'    not in variables)
    a2     = pymc.Uniform('a2', 0.8,1.2 ,   value = Sfid_params_5broadband5Gaussian['a2'],   observed = 'a2'    not in variables)
    a3     = pymc.Uniform('a3', 0.8,1.2 ,   value = Sfid_params_5broadband5Gaussian['a3'],   observed = 'a3'    not in variables)
    a4     = pymc.Uniform('a4', 0.8,1.2 ,   value = Sfid_params_5broadband5Gaussian['a4'],   observed = 'a4'    not in variables)
    sigma0 = pymc.Uniform('sigma0', 0.0,20.0,   value = Sfid_params_5broadband5Gaussian['sigma0'],   observed = 'sigma0'    not in variables)
    sigma1 = pymc.Uniform('sigma1', 0.0,20.0,   value = Sfid_params_5broadband5Gaussian['sigma1'],   observed = 'sigma1'    not in variables)
    sigma2 = pymc.Uniform('sigma2', 0.0,20.0,   value = Sfid_params_5broadband5Gaussian['sigma2'],   observed = 'sigma2'    not in variables)
    sigma3 = pymc.Uniform('sigma3', 0.0,20.0,   value = Sfid_params_5broadband5Gaussian['sigma3'],   observed = 'sigma3'    not in variables)
    sigma4 = pymc.Uniform('sigma4', 0.0,20.0,   value = Sfid_params_5broadband5Gaussian['sigma4'],   observed = 'sigma4'    not in variables)
    Alpha00= pymc.Uniform('Alpha00', -0.02,0.02,   value = Sfid_params_5broadband5Gaussian['Alpha00'],   observed = 'Alpha00'    not in variables)
    Alpha01= pymc.Uniform('Alpha01', -0.02,0.02,   value = Sfid_params_5broadband5Gaussian['Alpha01'],   observed = 'Alpha01'    not in variables)
    Alpha02= pymc.Uniform('Alpha02', -0.02,0.02,   value = Sfid_params_5broadband5Gaussian['Alpha02'],   observed = 'Alpha02'    not in variables)
    Alpha03= pymc.Uniform('Alpha03', -0.02,0.02,   value = Sfid_params_5broadband5Gaussian['Alpha03'],   observed = 'Alpha03'    not in variables)
    Alpha04= pymc.Uniform('Alpha04', -0.02,0.02,   value = Sfid_params_5broadband5Gaussian['Alpha04'],   observed = 'Alpha04'    not in variables)
    Alpha10= pymc.Uniform('Alpha10', -5.0,5.0,   value = Sfid_params_5broadband5Gaussian['Alpha10'],   observed = 'Alpha10'    not in variables)
    Alpha11= pymc.Uniform('Alpha11', -5.0,5.0,   value = Sfid_params_5broadband5Gaussian['Alpha11'],   observed = 'Alpha11'    not in variables)
    Alpha12= pymc.Uniform('Alpha12', -5.0,5.0,   value = Sfid_params_5broadband5Gaussian['Alpha12'],   observed = 'Alpha12'    not in variables)
    Alpha13= pymc.Uniform('Alpha13', -5.0,5.0,   value = Sfid_params_5broadband5Gaussian['Alpha13'],   observed = 'Alpha13'    not in variables)
    Alpha14= pymc.Uniform('Alpha14', -5.0,5.0,   value = Sfid_params_5broadband5Gaussian['Alpha14'],   observed = 'Alpha14'    not in variables)
    Alpha20= pymc.Uniform('Alpha20', 50.0,300.0,   value = Sfid_params_5broadband5Gaussian['Alpha20'],   observed = 'Alpha20'    not in variables)
    Alpha21= pymc.Uniform('Alpha21', 50.0,300.0,   value = Sfid_params_5broadband5Gaussian['Alpha21'],   observed = 'Alpha21'    not in variables)
    Alpha22= pymc.Uniform('Alpha22', 50.0,300.0,   value = Sfid_params_5broadband5Gaussian['Alpha22'],   observed = 'Alpha22'    not in variables)
    Alpha23= pymc.Uniform('Alpha23', 50.0,300.0,   value = Sfid_params_5broadband5Gaussian['Alpha23'],   observed = 'Alpha23'    not in variables)
    Alpha24= pymc.Uniform('Alpha24', 50.0,300.0,   value = Sfid_params_5broadband5Gaussian['Alpha24'],   observed = 'Alpha24'    not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, b0=b0,b1=b1,b2=b2,b3=b3,b4=b4,a0=a0,a1=a1,a2=a2,a3=a3,a4=a4,sigma0=sigma0,sigma1=sigma1,sigma2=sigma2,sigma3=sigma3,sigma4=sigma4,Alpha00=Alpha00,Alpha01=Alpha01,Alpha02=Alpha02,Alpha03=Alpha03,Alpha04=Alpha04,Alpha10=Alpha10,Alpha11=Alpha11,Alpha12=Alpha12,Alpha13=Alpha13,Alpha14=Alpha14,Alpha20=Alpha20,Alpha21=Alpha21,Alpha22=Alpha22,Alpha23=Alpha23,Alpha24=Alpha24):
        ll=0.
        pars = np.array([b0,b1,b2,b3,b4,a0,a1,a2,a3,a4,
          sigma0,
          sigma1,
          sigma2,
          sigma3,
          sigma4,
          Alpha00,
          Alpha01,
          Alpha02,
          Alpha03,
          Alpha04,
          Alpha10,
          Alpha11,
          Alpha12,
          Alpha13,
          Alpha14,
          Alpha20,
          Alpha21,
          Alpha22,
          Alpha23,
          Alpha24
          ])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_broadbandGaussian = {
               'b0':0.005,

                'a0':1.0,

                'sigma0':10.0,

                'Alpha00':0.008,

                'Alpha10':-2.5,

                'Alpha20':150.0,
                }

def Sll_model_broadbandGaussian(datasets, variables = ['b0','a0','sigma0','Alpha00','Alpha10','Alpha20'], fidvalues = Sfid_params_broadbandGaussian):
    """ 
      xi(r;z) = Sum_z bz**2* exp(-0.5*(r - az*get_rs() )**2/(sigmaz)**2 ) + Alpha0z + Alpha1z/r + Alpha2z/r**2 
      conected with rpymc_xi_ALLz_aiso.py
      one redshift bin
    """
    if (isinstance(datasets, list) is False): datasets=[datasets]
    # 0.000,0.01 or 1.6,2.2
    b0     = pymc.Uniform('b0', 0.000,0.01, value = Sfid_params_broadbandGaussian['b0'], observed = 'b0' not in variables)
    # 0.8,1.2
    a0     = pymc.Uniform('a0', 0.8,1.2 ,   value = Sfid_params_broadbandGaussian['a0'],   observed = 'a0'    not in variables)
    # 0.0,20.0
    sigma0 = pymc.Uniform('sigma0', 5.0,15.0,   value = Sfid_params_broadbandGaussian['sigma0'],   observed = 'sigma0'    not in variables)
    # -0.02,0.02
    Alpha00= pymc.Uniform('Alpha00', -0.02,0.02,   value = Sfid_params_broadbandGaussian['Alpha00'],   observed = 'Alpha00'    not in variables)
    # -5.0,5.0
    Alpha10= pymc.Uniform('Alpha10', -5.0,5.0,   value = Sfid_params_broadbandGaussian['Alpha10'],   observed = 'Alpha10'    not in variables)
    # 50.0,300.0
    Alpha20= pymc.Uniform('Alpha20', 50.0,300.0,   value = Sfid_params_broadbandGaussian['Alpha20'],   observed = 'Alpha20'    not in variables)

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, b0=b0,a0=a0,sigma0=sigma0,Alpha00=Alpha00,Alpha10=Alpha10,Alpha20=Alpha20):
        ll=0.
        pars = np.array([b0,
          a0,
          sigma0,
          Alpha00,
          Alpha10,
          Alpha20,
          ])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_omol = {
                'om':0.3156,                                                      
                'ol':1.-0.3156,
                }

def Sll_model_omol(datasets, variables = ['om','ol'], fidvalues = Sfid_params_omol):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    om        = pymc.Uniform('om', 0.0,1.0 ,  value = Sfid_params_omol['om'],   observed = 'om'     not in variables)
    ol        = pymc.Uniform('ol', 0.0,3.0,   value = Sfid_params_omol['ol'],   observed = 'ol'     not in variables) # 0.090,0.300                                                                
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, om=om,ol=ol):
        ll=0.
        pars = np.array([om,ol]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())



Sfid_params_omolhrd = {
                'om':0.3156,                                                      
                'ol':1.-0.3156,
                'hrd':0.6727*147.2
                }

def Sll_model_omolhrd(datasets, variables = ['om','ol','hrd'], fidvalues = Sfid_params_omolhrd):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    om        = pymc.Uniform('om', 0.0,1.0 ,     value = Sfid_params_omolhrd['om'] ,  observed = 'om'  not in variables)
    ol_min,ol_max=-1.0,2.0
    ol        = pymc.Uniform('ol',  ol_min,ol_max,     value = Sfid_params_omolhrd['ol']    ,  observed = 'ol'  not in variables) 
    print ol_min,ol_max
    #hrd       = pymc.Uniform('hrd', 40.0,160.0,   value = Sfid_params_omolhrd['hrd'],  observed = 'hrd' not in variables) 
    hrd_min,hrd_max=98.0,100.0 # 98.0,100.0 #strong 
    hrd       = pymc.Uniform('hrd', hrd_min,hrd_max,   value = Sfid_params_omolhrd['hrd'],  observed = 'hrd' not in variables) # Strong Prior Suggested by J Rich
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, om=om,ol=ol,hrd=hrd):
        ll=0.
        pars = np.array([om,ol,hrd]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_omolhrh = {
                'om':0.3156,                                                      
                'ol':1.-0.3156,
                'hrh':0.6727*92.99064263
                }

def Sll_model_omolhrh(datasets, variables = ['om','ol','hrh'], fidvalues = Sfid_params_omolhrd):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    om        = pymc.Uniform('om', 0.0,1.0 ,     value = Sfid_params_omolhrh['om'] ,  observed = 'om'  not in variables)
    ol        = pymc.Uniform('ol', 0.0,1.0 ,     value = Sfid_params_omolhrh['ol']    ,  observed = 'ol'  not in variables) 
    #hrd       = pymc.Uniform('hrd', 40.0,160.0,   value = Sfid_params_omolhrd['hrd'],  observed = 'hrd' not in variables) 
    hrh       = pymc.Uniform('hrh', 61.5,63.5,   value = Sfid_params_omolhrh['hrh'],  observed = 'hrh' not in variables) # Strong Prior Suggested by J Rich
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, om=om,ol=ol,hrd=hrh):
        ll=0.
        pars = np.array([om,ol,hrh]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_omolh = {
                'om':0.3156,                                                      
                'ol':1.-0.3156,
                'h':0.6727
                }

def Sll_model_omolh(datasets, variables = ['om','ol','h'], fidvalues = Sfid_params_omolh):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    om        = pymc.Uniform('om', 0.0,1.0 ,     value = Sfid_params_omolh['om'] ,  observed = 'om'  not in variables)
    ol        = pymc.Uniform('ol', 0.0,1.0 ,     value = Sfid_params_omolh['ol']    ,  observed = 'ol'  not in variables) 
    #hrd       = pymc.Uniform('hrd', 40.0,160.0,   value = Sfid_params_omolhrd['hrd'],  observed = 'hrd' not in variables) 
    h         = pymc.Uniform('h', 0.67,0.68  ,     value = Sfid_params_omolh['h'],  observed = 'h' not in variables) # Strong Prior Suggested by J Rich
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, om=om,ol=ol,h=h):
        ll=0.
        pars = np.array([om,ol,h]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


"""
Sfid_params_ABOmOl = {
               'A': 2.2,
               'B': 0.5,
               'Om': 0.31,
               'Ol': 0.69,
                }

def Sll_model_ABOmOl(datasets, variables = ['A','B','Omega_m','Omega_Lambda'], fidvalues = Sfid_params_ABOmOl):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    A     = pymc.Uniform('A', 1.0,3.0, value = Sfid_params_ABOmOl['A'], observed = 'A' not in variables)
    B     = pymc.Uniform('B', 0.4,0.7, value = Sfid_params_ABOmOl['B'], observed = 'B' not in variables)
    Om     = pymc.Uniform('Om', 0.0,1.0, value = Sfid_params_ABOmOl['Om'], observed = 'Om' not in variables)
    Ol     = pymc.Uniform('Ol', 0.0,3.0, value = Sfid_params_ABOmOl['Ol'], observed = 'Ol' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, A=A,B=B,Om=Om,Ol=Ol):
        ll=0.
        pars = np.array([A,B,Om,Ol])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())
"""

Sfid_params_omol_hrd_AB = {
               'om': 0.31,
               'ol': 0.69,
               'hrd':0.6727*147.2,
               'A': 2.2,
               'B': 0.55,
                }

def Sll_model_omol_hrd_AB(datasets, variables = ['om','ol','hrd','A','B'], fidvalues = Sfid_params_omol_hrd_AB):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    om     = pymc.Uniform('om', 0.0,1.0, value = Sfid_params_omol_hrd_AB['om'], observed = 'om' not in variables)
    ol     = pymc.Uniform('ol', 0.0,3.0, value = Sfid_params_omol_hrd_AB['ol'], observed = 'ol' not in variables)
    hrd_min,hrd_max=98.0,100.0 # 98.0,100.0 #strong 
    hrd    = pymc.Uniform('hrd', hrd_min,hrd_max,   value = Sfid_params_omol_hrd_AB['hrd'],  observed = 'hrd' not in variables) # Strong Prior Suggested by J Rich
    A_min,A_max=2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0
    B_min,B_max=0.545,0.555     # optimum!   
    A      = pymc.Uniform('A', A_min,A_max, value = Sfid_params_omol_hrd_AB['A'], observed = 'A' not in variables)
    B      = pymc.Uniform('B', B_min,B_max, value = Sfid_params_omol_hrd_AB['B'], observed = 'B' not in variables)

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, om=om,ol=ol,hrd=hrd,A=A,B=B):
        ll=0.
        pars = np.array([om,ol,hrd,A,B])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_ABomolhrh = {
               'A': 2.0,
               'B': 0.5,
               'om': 0.31,
               'ol': 0.69,
               'hrh':62.518571258448226,
                }

def Sll_model_ABomolhrh(datasets, variables = ['A','B','om','ol','hrh'], fidvalues = Sfid_params_ABomolhrh):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    A      = pymc.Uniform('A', 1.0,3.0, value = Sfid_params_ABomolhrh['A'], observed = 'A' not in variables)
    B      = pymc.Uniform('B', 0.4,0.7, value = Sfid_params_ABomolhrh['B'], observed = 'B' not in variables)
    om     = pymc.Uniform('om', 0.0,1.0, value = Sfid_params_ABomolhrh['om'], observed = 'om' not in variables)
    ol     = pymc.Uniform('ol', 0.0,3.0, value = Sfid_params_ABomolhrh['ol'], observed = 'ol' not in variables)
    hrh_min,hrh_max=50,70 #61,63 #strong 
    hrh    = pymc.Uniform('hrh', hrh_min,hrh_max,   value = Sfid_params_ABomolhrh['hrh'],  observed = 'hrh' not in variables) # Strong Prior Suggested by J Rich
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, A=A,B=B,om=om,ol=ol,hrh=hrh,):
        ll=0.
        pars = np.array([A,B,om,ol,hrh])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_ABomol = {
               'A':  2.1, #2.2,
               'B':  0.55,#0.55,
               'om': 0.31,
               'ol': 0.69,
                }

def Sll_model_ABomol(datasets, variables = ['A','B','om','ol'], fidvalues = Sfid_params_ABomol):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    
    #A_min,A_max=0.0,4.0 #1.6,2.4 #1.6,2.4#1.5,2.8#2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0 A1.6_2.4_B0.545_0.555
    #B_min,B_max=0.0,1.5  #0.545,0.555     # optimum!   
    A_min,A_max=0.0,3.0 #1.6,2.4 #1.6,2.4#1.5,2.8#2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0 A1.6_2.4_B0.545_0.555
    B_min,B_max=-1,3.0  #0.545,0.555     # optimum!   
    A      = pymc.Uniform('A', A_min,A_max, value = Sfid_params_ABomol['A'], observed = 'A' not in variables)
    B      = pymc.Uniform('B', B_min,B_max, value = Sfid_params_ABomol['B'], observed = 'B' not in variables)
    om     = pymc.Uniform('om', 0.0,1.0, value = Sfid_params_ABomol['om'], observed = 'om' not in variables)
    ol     = pymc.Uniform('ol', 0.0,1.0, value = Sfid_params_ABomol['ol'], observed = 'ol' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, A=A,B=B,om=om,ol=ol):
        ll=0.
        pars = np.array([A,B,om,ol])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_AB = {
               'A': 2.0, #1.5, #2.2,
               'B': 0.55,
               #'om': 0.31,
               #'ol': 0.69,
                }

def Sll_model_AB(datasets, variables = ['A','B'], fidvalues = Sfid_params_AB):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    
    #A_min,A_max=1.5,2.8#2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0
    #B_min,B_max=-2,1.  #0.545,0.555     # optimum!   

    #A_min,A_max=1.6,2.4 #1.0,2.0#2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0
    #B_min,B_max=0.0,2.0 #0.545,0.555     # optimum!   

    A_min,A_max=0.0,3.0 
    B_min,B_max=-1,3.0 

    A     = pymc.Uniform('A', A_min,A_max, value = Sfid_params_AB['A'], observed = 'A' not in variables)
    B     = pymc.Uniform('B', B_min,B_max, value = Sfid_params_AB['B'], observed = 'B' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, A=A,B=B):
        ll=0.
        pars = np.array([A,B,])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_A = {
               'A': 1.5, #1.5, #2.2,
               #'om': 0.31,
               #'ol': 0.69,
                }

def Sll_model_A(datasets, variables = ['A'], fidvalues = Sfid_params_A):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    
    #A_min,A_max=1.5,2.8#2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0
    #B_min,B_max=-2,1.  #0.545,0.555     # optimum!   

    #A_min,A_max=1.6,2.4 #1.0,2.0#2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0
    #B_min,B_max=0.0,2.0 #0.545,0.555     # optimum!   

    A_min,A_max=0.5,2.0 
    A     = pymc.Uniform('A', A_min,A_max, value = Sfid_params_AB['A'], observed = 'A' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, A=A):
        ll=0.
        pars = np.array([A,])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_Aomol = {
               'A': 1.2, # 1.2
               #'B': 0.57, # 0.55
               'om': 0.31,
               'ol': 0.69,
                }

def Sll_model_Aomol(datasets, variables = ['A','om','ol'], fidvalues = Sfid_params_Aomol):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    A_min,A_max=0.5,4.0 # optimum! #2.15,2.25 still shit # 1.0,3.0
    om_min,om_max = 0.2,0.90 #0.0,1.0 but for interpolated Rh we need 0.2,1.0
    ol_min,ol_max = 0.4,0.90 #0.0,1.0 but for interpolated Rh we need 0.4,1.0
    A     = pymc.Uniform('A', A_min,A_max, value = Sfid_params_Aomol['A'], observed = 'A' not in variables)
    om     = pymc.Uniform('om', om_min,om_max, value = Sfid_params_Aomol['om'], observed = 'om' not in variables)
    ol     = pymc.Uniform('ol', ol_min,ol_max, value = Sfid_params_Aomol['ol'], observed = 'ol' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, A=A,om=om,ol=ol):
        ll=0.
        pars = np.array([A,om,ol])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_Aom = {
               'A': 1.2, # 1.2
               'om': 0.31,
                }

def Sll_model_Aom(datasets, variables = ['A','om'], fidvalues = Sfid_params_Aom):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    A_min,A_max=0.5,2.0 # optimum! #2.15,2.25 still shit # 1.0,3.0
    A      = pymc.Uniform('A', A_min,A_max, value = Sfid_params_Aom['A'] , observed = 'A' not in variables)
    om     = pymc.Uniform('om', 0.0,   1.0, value = Sfid_params_Aom['om'], observed = 'om' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, A=A,om=om):
        ll=0.
        pars = np.array([A,om])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_okom = {
               'ok': -0.01, # 1.2
               'om': 0.31,
                }

def Sll_model_okom(datasets, variables = ['ok','om'], fidvalues = Sfid_params_okom):
    if (isinstance(datasets, list) is False): datasets=[datasets]
  
    ok     = pymc.Uniform('ok', -1.0,   1.0, value = Sfid_params_okom['ok'], observed = 'ok' not in variables)
    om     = pymc.Uniform('om',  0.0,   1.0, value = Sfid_params_okom['om'], observed = 'om' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, om=om,ok=ok):
        ll=0.
        pars = np.array([ok,om])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_b0omgammamg = {
               'b0': 1.2, # 1.2
               'om': 0.31,
               'gammamg':0.55,
                }

def Sll_model_b0omgammamg(datasets, variables = ['b0','om','gammamg'], fidvalues = Sfid_params_b0omgammamg):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    b0_min,b0_max=0.0,2.0 # optimum! #2.15,2.25 still shit # 1.0,3.0
    b0     = pymc.Uniform('b0'   , b0_min,b0_max, value = Sfid_params_b0omgammamg['b0'], observed = 'b0' not in variables)
    om     = pymc.Uniform('om'   , 0.2,0.6      , value = Sfid_params_b0omgammamg['om'], observed = 'om' not in variables)
    gammamg= pymc.Uniform('gammamg', 0.0,1.0    , value = Sfid_params_b0omgammamg['gammamg'], observed = 'gammamg' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, b0=b0,om=om,gammamg=gammamg):
        ll=0.
        pars = np.array([b0,om,gammamg])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_omolAB = {
               'om': 0.31,
               'ol': 0.69,
               #'A':  2.1, #2.2,
               #'B':  0.55,#0.55,
               'A':0.8,
               'B':0.2,
                }

def Sll_model_omolAB(datasets, variables = ['om','ol','A','B'], fidvalues = Sfid_params_omolAB):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    
    #A_min,A_max=0.0,4.0 #1.6,2.4 #1.6,2.4#1.5,2.8#2.19999,2.21111 # optimum! #2.15,2.25 still shit # 1.0,3.0 A1.6_2.4_B0.545_0.555
    #B_min,B_max=0.0,1.5  #0.545,0.555     # optimum!   
    A_min,A_max=-1.0,2.0 #0.0,3.0 
    B_min,B_max=-1.0,1.0 #0.80*0.5,1.20*0.5  
    om     = pymc.Uniform('om', 0.0,1.0, value = Sfid_params_omolAB['om'], observed = 'om' not in variables)
    ol     = pymc.Uniform('ol', 0.0,3.0, value = Sfid_params_omolAB['ol'], observed = 'ol' not in variables)
    A      = pymc.Uniform('A', A_min,A_max, value = Sfid_params_omolAB['A'], observed = 'A' not in variables)
    B      = pymc.Uniform('B', B_min,B_max, value = Sfid_params_omolAB['B'], observed = 'B' not in variables)

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, om=om,ol=ol,A=A,B=B):
        ll=0.
        pars = np.array([om,ol,A,B])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())



Sfid_params_omolb0b1b2 = {
               'om':0.31,
               'ol':0.69,
               'b0':0.1,
               'b1':0.1,
               'b2':2.0
                }

def Sll_model_omolb0b1b2(datasets, variables = ['om','ol','b0','b1','b2'], fidvalues = Sfid_params_omolb0b1b2):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    
    b0_min,b0_max=-1.0,1.0 #0.0,3.0 
    b1_min,b1_max=-1.0,1.0 #0.80*0.5,1.20*0.5  
    b2_min,b2_max=+0.0,3.0 #0.0,3.0 
    om  = pymc.Uniform('om', 0.0   ,   1.0, value = Sfid_params_omolb0b1b2['om'], observed = 'om' not in variables)
    ol  = pymc.Uniform('ol', 0.0   ,   3.0, value = Sfid_params_omolb0b1b2['ol'], observed = 'ol' not in variables)
    b0  = pymc.Uniform('b0', b0_min,b0_max, value = Sfid_params_omolb0b1b2['b0'], observed = 'b0' not in variables)
    b1  = pymc.Uniform('b1', b1_min,b1_max, value = Sfid_params_omolb0b1b2['b1'], observed = 'b1' not in variables)
    b2  = pymc.Uniform('b2', b2_min,b2_max, value = Sfid_params_omolb0b1b2['b2'], observed = 'b2' not in variables)

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, om=om,ol=ol,b0=b0,b1=b1,b2=b2):
        ll=0.
        pars = np.array([om,ol,b0,b1,b2])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_omolwb0 = {
               'om':0.31,
               'ol':0.69,
               'w': -1.0,
               'b0':1.5,
                }

def Sll_model_omolwb0(datasets, variables = ['om','ol','w','b0'], fidvalues = Sfid_params_omolwb0):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    
    b0_min,b0_max=0.0,3.0 #0.0,3.0 

    om  = pymc.Uniform('om', 0.0   ,   1.0, value = Sfid_params_omolwb0['om'], observed = 'om' not in variables)
    ol  = pymc.Uniform('ol', 0.0   ,   3.0, value = Sfid_params_omolwb0['ol'], observed = 'ol' not in variables)
    w   = pymc.Uniform('w', -2.0   ,   0.0, value = Sfid_params_omolwb0['w'], observed = 'w' not in variables)
    b0  = pymc.Uniform('b0', b0_min,b0_max, value = Sfid_params_omolwb0['b0'], observed = 'b0' not in variables)


    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, om=om,ol=ol,w=w,b0=b0):
        ll=0.
        pars = np.array([om,ol,w,b0])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


fid_BR = {
               'n': 0.12,
               'H0': 78.74, # 0.55
               'Om': 0.31,
               'OX': 0.69,
                }


def ll_BR(datasets, variables = ['n', 'H0', 'Om', 'OX'], fidvalues = fid_BR):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    n     = pymc.Uniform('n', 0.0,0.3, value = fid_BR['n'], observed = 'n' not in variables)
    H0     = pymc.Uniform('H0', 50.0,100.0, value = fid_BR['H0'], observed = 'H0' not in variables)
    Om     = pymc.Uniform('Om', 0.0,1.0, value = fid_BR['Om'], observed = 'Om' not in variables)
    OX     = pymc.Uniform('OX', 0.0,3.0, value = fid_BR['OX'], observed = 'OX' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, n=n,H0=H0,Om=Om,OX=OX):
        ll=0.
        pars = np.array([n,H0,Om,OX])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

fid_BR_nOm = {
               'n': 0.12,
               'Om': 0.31,
                }

def ll_BR_nOm(datasets, variables = ['n', 'Om'], fidvalues = fid_BR_nOm):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    n_min,n_max = -3.0,1.0 #Best-3.,1.0    # -1.5,3. fit the 2008 J. Larena 0808.1161v2 fig 2
    Om_min,Om_max = 0.0, 1.0#Best-0.5,0.7 # 0.0,0.7 

    n      = pymc.Uniform('n' , n_min,n_max , value = fid_BR_nOm['n'], observed = 'n' not in variables)
    Om     = pymc.Uniform('Om', Om_min,Om_max, value = fid_BR_nOm['Om'], observed = 'Om' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, n=n,Om=Om):
        ll=0.
        pars = np.array([n,Om])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


fid_BR_nOmOX = {
               'n':  0.12,
               'Om': 0.31,
               'OX': 0.1,
                }

def ll_BR_nOmOX(datasets, variables = ['n', 'Om', 'OX'], fidvalues = fid_BR_nOmOX):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    n_min,n_max   = -1.5, 0.5 #-1.0, 0.5 
    Om_min,Om_max = 0.0, 1.0 #+0.0, 1.0 
    OX_min,OX_max = 0.0, 1.0 #+0.0, 1.0 

    n      = pymc.Uniform('n' , n_min,n_max  , value = fid_BR_nOmOX['n'] , observed = 'n'  not in variables)
    Om     = pymc.Uniform('Om', Om_min,Om_max, value = fid_BR_nOmOX['Om'], observed = 'Om' not in variables)
    OX     = pymc.Uniform('OX', OX_min,OX_max, value = fid_BR_nOmOX['OX'], observed = 'OX' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, n=n,Om=Om,OX=OX):
        ll=0.
        pars = np.array([n,Om,OX])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

fid_test_linear = {
               'a': 1.0,
               'b': 0.0,
                }

def ll_test_linear(datasets, variables = ['a', 'b'], fidvalues = fid_test_linear):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    a      = pymc.Uniform('a', -4.,4.,   value = fid_test_linear['a'], observed = 'a' not in variables)
    b      = pymc.Uniform('b', -10.0,40.0, value = fid_test_linear['b'], observed = 'b' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, a=a,b=b):
        ll=0.
        pars = np.array([a,b])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

fid_test_linear_fixed_b = {
               'a': 2.0,
                }
def ll_test_linear_fixed_b(datasets, variables = ['a'], fidvalues = fid_test_linear_fixed_b):
    if (isinstance(datasets, list) is False): datasets=[datasets]
    a      = pymc.Uniform('a', 0.0,4.,   value = fid_test_linear_fixed_b['a'], observed = 'a' not in variables)
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, a=a):
        ll=0.
        pars = np.array([a])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_b0OmfNL = {
               'b0':5.0,
               #'h':0.67,
               'om0':0.3,
               'fNL':0.0,
                }

def Sll_model_b0OmfNL(datasets, variables = ['b0','om0','fNL'], fidvalues = Sfid_params_b0OmfNL):

    if (isinstance(datasets, list) is False): datasets=[datasets]
    b0     = pymc.Uniform('b0',    1.0,8.0 , value = Sfid_params_b0OmfNL['b0'] , observed = 'b0'  not in variables)
    om0     = pymc.Uniform('om0',    0.1,1.0 , value = Sfid_params_b0OmfNL['om0'] , observed = 'om0' not in variables) 
    fNL    = pymc.Uniform('fNL', -100.,100., value = Sfid_params_b0OmfNL['fNL'], observed = 'fNL' not in variables) 
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, b0=b0,om0=om0,fNL=fNL): 
        ll=0.
        pars = np.array([b0,om0,fNL]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())


Sfid_params_b0fNL = {
               'b0':2.0,
               #'h':0.67,
               'fNL':0.0,
                }

def Sll_model_b0fNL(datasets, variables = ['b0','fNL'], fidvalues = Sfid_params_b0fNL):

    if (isinstance(datasets, list) is False): datasets=[datasets]
    b0     = pymc.Uniform('b0',    1.0,8.0 , value = Sfid_params_b0fNL['b0'] , observed = 'b0'  not in variables)
    fNL    = pymc.Uniform('fNL', -200.,200., value = Sfid_params_b0fNL['fNL'], observed = 'fNL' not in variables) 
    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0, b0=b0,fNL=fNL): 
        ll=0.
        pars = np.array([b0,fNL]) 
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())

Sfid_params_dcabrsrVom = {
              'dc':-1.0,
              'a':2.3  ,
              'b':7.0  ,
              'rs':30  ,
              'rV':30  ,
              'om':0.32,
                }

def Sll_model_dcabrsrVom(datasets, variables = ['dc','a','b','rs','rV','om'], fidvalues = Sfid_params_dcabrsrVom):
    if (isinstance(datasets, list) is False): datasets=[datasets]
        
    dc = pymc.Uniform('dc', -3.,0.0 , value = Sfid_params_dcabrsrVom['dc'], observed = 'dc' not in variables)
    a  = pymc.Uniform('a' , 0.0,3.0 , value = Sfid_params_dcabrsrVom['a'] , observed = 'a' not in variables)
    b  = pymc.Uniform('b' , 6.0,10.0, value = Sfid_params_dcabrsrVom['b'] , observed = 'b' not in variables)
    rs = pymc.Uniform('rs', 10.,100., value = Sfid_params_dcabrsrVom['rs'], observed = 'rs' not in variables)
    rV = pymc.Uniform('rV', 10.,100., value = Sfid_params_dcabrsrVom['rV'], observed = 'rV' not in variables)
    om = pymc.Uniform('om', 0.0,2.0 , value = Sfid_params_dcabrsrVom['om'], observed = 'om' not in variables)

    @pymc.stochastic(trace=True,observed=True,plot=False)
    def loglikelihood(value=0,dc=dc,a=a,b=b,rs=rs,rV=rV,om=om ):
        ll=0.
        pars = np.array([dc,a,b,rs,rV,om])
        for ds in datasets:
            ll=ll+ds(pars)
        return(ll)
    return(locals())
'''
