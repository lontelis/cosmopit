import numpy as np
from pylab import *
from scipy import interpolate,linalg

# taken from JC Hamilton and remodified by P.Ntelis 2014

class MySplineFitting:
    def __init__(self,xin,yin,covarin,nbspl,nmock_prec=None,logspace=False,invcov=None,cholesky=False):
        # input parameters
        self.x=xin
        self.y=yin
        self.nbspl=nbspl
        covar=covarin
        if np.size(np.shape(covarin)) == 1:
            err=covarin
            covar=np.zeros((np.size(err),np.size(err)))
            covar[np.arange(np.size(err)),np.arange(np.size(err))]=err**2

        self.covar=covar
        if invcov is None:
            self.invcovar = np.linalg.inv(covar)
        else:
            self.invcovar = invcov

        if nmock_prec!=None:
            precision_correction =  ( nmock_prec-len(xin)-2. ) / ( nmock_prec - 1. )
            self.invcovar = self.invcovar* precision_correction

        # Prepare splines
        xspl=np.linspace(np.min(self.x),np.max(self.x),nbspl)
        if logspace==True: xspl=np.logspace(np.log10(np.min(self.x)),np.log10(np.max(self.x)),nbspl)
        self.xspl=xspl
        F=np.zeros((np.size(xin),nbspl))
        self.F=F
        for i in np.arange(nbspl):
            self.F[:,i]=self.get_spline_tofit(xspl,i,xin)

        # solution of the chi square
        if cholesky:
            U_cho = linalg.cho_factor(covar)
            ft_cinv_y=np.dot(np.transpose(self.F),linalg.cho_solve(U_cho, self.y))
            covout = np.linalg.inv(np.dot(np.transpose(self.F),linalg.cho_solve(U_cho, self.F)))
        else:
            ft_cinv_y=np.dot(np.transpose(self.F),np.dot(self.invcovar,self.y))
            covout = np.array(np.matrix( np.dot(np.transpose(self.F),np.dot(self.invcovar,self.F)) ).I)
        alpha=np.dot(covout,ft_cinv_y)
        self.fitted=np.dot(self.F,alpha)
        
        # output
        self.residuals=self.y-self.fitted
        if cholesky:
            self.chi2=np.dot(np.transpose(self.residuals), linalg.cho_solve(U_cho, self.residuals))
        else:
            self.chi2=np.dot(np.transpose(self.residuals), np.dot(self.invcovar, self.residuals))
        self.ndf=np.size(xin)-np.size(alpha)
        self.alpha=alpha
        self.covout=covout
        self.dalpha=np.sqrt(np.diagonal(covout))

    
    def __call__(self,x):
        theF=np.zeros((np.size(x),self.nbspl))
        for i in np.arange(self.nbspl): theF[:,i]=self.get_spline_tofit(self.xspl,i,x)
        return(dot(theF,self.alpha))

    def with_alpha(self,x,alpha):
        theF=np.zeros((np.size(x),self.nbspl))
        for i in np.arange(self.nbspl): theF[:,i]=self.get_spline_tofit(self.xspl,i,x)
        return(dot(theF,alpha))
            
    def get_spline_tofit(self,xspline,index,xx):
        yspline=zeros(np.size(xspline))
        yspline[index]=1.
        tck=interpolate.splrep(xspline,yspline)
        yy=interpolate.splev(xx,tck,der=0)
        return(yy)
