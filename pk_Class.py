from pylab import *
import numpy as np
from numpy import *
from scipy import interpolate
from scipy import integrate
from scipy import special
import numpy
from classy import Class
from cosmopit import cosmology
# Taken from J.C.Hamilton and remodified by P.Ntelis June 2014

class theory:
    '''
    A software to compute the theoretical Power Spectrum from CLASS Software and each Derivatives, 
    2-Point Correlation function, Fractal Correlation Dimension, Homogeneity Scale
    P(k,z), xi(r,z), D2(r,z), N(r,z), Rh(z)
    k  in Mpc^{-1}
    r  in h^{-1} Mpc
    Rh in h^{-1} Mpc
    Observables are reviewed on arXiv:1702.02159
    # Taken from J.C.Hamilton and remodified by P.Ntelis June 2014
    # Line 42 and 44 are ususfull because since CLASS uses sometimes the alternative definition of (sum_i Omega_i) = 1 + Omega_k. Be carefull on the output before applying.
    # In this analysis we use: $\Omega_k$ + $\Omega_m$ + $\Omega_{\Lambda}$ = 1
    '''
    def __init__(self,cosmopars,z=None,h_fiducial_xith_norm=0.6727,sig8_Ext=None,kmax=20.,kmin=0.001,nk=50000,halofit=False,ExtraPars=False,P_k_max_h=100.,z_max_pk=5.): # kmax=20.
        cosmo_CLASS = Class()
        extra_CLASS_pars={
            'non linear': '',
            'output': 'mPk mTk', # Add the request for total matter transfer function in fourier space
            'z_max_pk': z_max_pk, #10.
            #'k_pivot' : 0.05,
            'P_k_max_h/Mpc': P_k_max_h} #100.
        if 'Omega_k' in cosmopars.keys(): pass 
        else: cosmopars.update({'Omega_k':0.0})
        if ( (cosmopars['Omega_k'] < 1e-10) and (cosmopars['Omega_k'] > -1e-10) ) :  cosmopars['Omega_k']=0.0
        #if 'Omega_Lambda' in cosmopars.keys(): print cosmopars['Omega_Lambda']

        cosmopars.update(extra_CLASS_pars)
        if halofit: cosmopars['non linear']='halofit'
        
        cosmo_CLASS.set(cosmopars)
        #cosmo_CLASS.pars['Omega_k'] = -cosmo_CLASS.pars['Omega_k']  ## #need for alternative definition of Ok in CLASS
        cosmo_CLASS.compute()
        #cosmo_CLASS.pars['Omega_k'] = -cosmo_CLASS.pars['Omega_k'] ## #need for alternative definition of Ok in CLASS
        
        self.h_fiducial_xith_norm=h_fiducial_xith_norm
        self.r_s = cosmo_CLASS.rs_drag()
        self.h   = cosmopars['h']
        self.Omega_m=(cosmo_CLASS.pars['omega_b']+cosmo_CLASS.pars['omega_cdm'])/(cosmo_CLASS.pars['h']**2) 
        self.n_s    = cosmo_CLASS.pars['n_s']

        if ExtraPars: self.w0,self.wa=cosmo_CLASS.pars['w0_fld'],cosmo_CLASS.pars['wa_fld']
        else:         self.w0,self.wa=-1.,0.

        self.Omega_k = cosmo_CLASS.pars['Omega_k']
        self.Omega_L = 1. - self.Omega_m - self.Omega_k

        if np.isclose(self.Omega_L,cosmo_CLASS.Omega_Lambda(),rtol=3):
            print('Check the definition of Omega_k in CLASS')
            print('https://github.com/lesgourg/class_public/wiki/Installation')
            print('Omega_k + sum_i Omega_i = 1')
            print('self.Omega_L','cosmo_CLASS.Omega_Lambda()' )
            print(self.Omega_L,cosmo_CLASS.Omega_Lambda() )

        #if not np.isclose(self.Omega_k+self.Omega_L+self.Omega_m , 1.0,atol=1e-05) : 
        #    raise NameError('Ok+OL+Om==1 not fullfilled')
        #print 'Omegak Omega_L Omega_m |self'
        #print self.Omega_k,self.Omega_L,self.Omega_m
        #print 'Omegak Omega_L Omega_m |cosmopars'
        #print cosmopars['Omega_k'],cosmopars['Omega_Lambda'],self.Omega_m
        #print 'Sum Omega = %0.1f'%(self.Omega_k+self.Omega_L+self.Omega_m)

        sig8_CLASS = cosmo_CLASS.sigma8()
        self.sig8 = sig8_CLASS
        
        self.nk=nk
        self.k=np.linspace(kmin,kmax,nk) #0.001
        self.kmin=kmin
        self.pk=self.k*0.
        self.sig8_Ext=sig8_Ext
        self.sig8_CLASS=sig8_CLASS
        self.cosmo_CLASS=cosmo_CLASS

        self.redshift=z

        self.pk_class_MM,self.pk_interpol=self.calc_pk_CMMpk_interp_sig8(self.redshift)
        self.pk_class_MM0,self.pk_interpol0=self.calc_pk_CMMpk_interp_sig8(0.0)

        # compute Tk from class
        self.d_Transfer_tot,self.k_h_Mpc_for_transfer = cosmo_CLASS.get_transfer()['d_tot'],cosmo_CLASS.get_transfer()['k (h/Mpc)']

        #cosmo_CLASS.struct_cleanup()
        #cosmo_CLASS.empty()

    def __call__(self,k,*args, **kw):
        ''' 
            argument k: array of the wavenumbers
            returns an array of the 1D the matter power spectrum P(k) 
        '''
        return(self.pk_interpol(k))

    def calc_pk_CMMpk_interp_sig8(self,z):
        ''' calculating the pk from z and pk '''
        for i in np.arange(self.k.size):
            self.pk[i] = self.cosmo_CLASS.pk(self.k[i],z=z)
        #self.pk[0]=self.pk[1]
        if  self.sig8_Ext!=None:
            self.pk=self.pk * (  self.sig8_Ext / self.sig8_CLASS )**2 #//kaiser of using A_s
            self.sig8 = self.sig8_Ext
        
        pk_CMM=self.pk
        pk_interp=interpolate.interp1d(self.k,pk_CMM)#,bounds_error=False)
        return pk_CMM,pk_interp 

    def get_Tk(self,kin):
        r"""
        Forked from : https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/cosmology/power/transfers.html#CLASS.__call__
        Return the CLASS linear transfer function at :attr:`redshift`. This
        computes the transfer function from the CLASS linear power spectrum
        using:

        .. math::

            T(k) = \left [ P_L(k) / k^n_s  \right]^{1/2}.

        We normalize the transfer function :math:`T(k)` to unity as
        :math:`k \rightarrow 0` at :math:`z=0`.

        Parameters
        ---------
        k : float, array_like
            the wavenumbers in units of :math:`h \mathrm{Mpc}^{-1}`

        Returns
        -------
        Tk : float, array_like
            the transfer function evaluated at ``k``, normalized to unity on
            large scales

        """
        print('get_Tk still needs debugging')

        k = numpy.asarray(kin)
        nonzero = k>0
        linearP = self.pk_interpol0(k[nonzero]) 
        primordialP = k[nonzero]**(self.n_s) # put k into h Mpc^{-1}

        # find the low-k amplitude to normalize to 1 as k->0 at z = 0
        self._norm = 1.

        # return shape
        Tk = numpy.ones(nonzero.shape)

        # at k=0, this is 1.0 * D(z), where T(k) = 1.0 at z=0
        Tk[~nonzero] = 1.0

        # fill in all k>0
        Tk[nonzero] = self._norm * (linearP / primordialP)**0.5

        return Tk

    def get_P_at_k(self,k_specific):
        return self.cosmo_CLASS.pk(k_specific,z=self.redshift)

    def get_rs(self):
        """ returns the scale to the sound horizon on drag epoch"""
        return self.r_s

    def get_h(self):
        return self.h

    def get_Omega_k(self):
        return self.Omega_k

    def get_Omega_m(self):
        return self.Omega_m

    def get_Omega_L(self):
        return self.Omega_L

    def get_w0(self):
        return self.w0

    def get_wa(self):
        return self.wa
    
    def get_ns(self):
        return self.n_s

    def get_sig8(self):
        return self.sig8

    def get_redshift(self):
        return self.redshift 

    def set_redshift(self,z_input):
        self.redshift=z_input
        self.pk_class_MM,self.pk_interpol=self.calc_pk_CMMpk_interp_sig8(self.redshift)

    def empty_memory(self):
        self.cosmo_CLASS.struct_cleanup()
        self.cosmo_CLASS.empty()

    def pk_NL(self,x,bias=np.nan,sigp=None,kaiser=False,damping=False,galaxy=False):
        pk_input    = self.pk_interpol(x) 
        self.bias   =bias
        self.sigp   =sigp
        self.kaiser =kaiser
        self.damping=damping
        if kaiser:           #!# SOS
            #print 'did kaiser'
            Factor_NL = self.NL_factor(h=self.h,Omega0_m=self.Omega_m,z=self.redshift,bias=self.bias,sigp=self.sigp,nk=self.nk,k_theory=self.k,damping=self.damping,kaiser=self.kaiser)
            self.Factor_non_linearities = Factor_NL
            if damping: pass
            pk_input = self.pk_interpol(x) * self.Factor_non_linearities
            print('Factor Kaiser=%0.3f'%(self.Factor_non_linearities))
        else: pass           #!# SOS
        if galaxy: pk_input = pk_input*self.bias**2
        return pk_input
        
    def fctShape(self,x,pars):
        return pars[0] + pars[1]/x**2 + pars[2]/(x**2) 

    def pk2correlation(self,h=0.7,nk=5000,k=None,pk_in=None,use_h_norm=True):
        pkk=k*pk_in
        r=2.*pi/np.max(k)*np.arange(nk)
        cric=-np.imag(fft.fft(pkk)/nk)/r/2/pi**2*np.max(k)
        cric[0]=cric[1]
        h_fiducial_xith_norm=self.h_fiducial_xith_norm
        if use_h_norm:
            twoPCF=interpolate.interp1d(r*self.h_fiducial_xith_norm,cric) #,bounds_error=False)
        else:
            twoPCF=interpolate.interp1d(r,cric) #,bounds_error=False)
        return(twoPCF)

    def xi(self,x,bias=np.nan,sigp=None,kaiser=False,damping=False,galaxy=False,vShape=None,use_h_norm=True,
                                                if_Pk_VOIDS=False,D=35.,number_of_voids=122907.,volume_in_Mpc_h_power_3=2000**3.,if_Gaussian_Damping=False,sig_G=None,kstar=0.2):
        pk_in = self.pk_NL(self.k,bias=bias,sigp=sigp,kaiser=kaiser,damping=damping,galaxy=galaxy)
        if if_Pk_VOIDS=='pk_exclusion':
            P_exclusion_input = P_exclusion_fnt(self.k,D=D,number_of_voids=number_of_voids,volume_in_Mpc_h_power_3=volume_in_Mpc_h_power_3,if_Gaussian_Damping=if_Gaussian_Damping,sig_G=sig_G,kstar=kstar)
            pk_in = P_exclusion_input 
        elif if_Pk_VOIDS=='pk_exclusion + pk': 
            P_exclusion_input = P_exclusion_fnt(self.k,D=D,number_of_voids=number_of_voids,volume_in_Mpc_h_power_3=volume_in_Mpc_h_power_3,if_Gaussian_Damping=if_Gaussian_Damping,sig_G=sig_G,kstar=kstar)
            pk_in = P_exclusion_input * pk_in 
        else: pass

        if (type(vShape)==tuple) or (type(vShape)==np.ndarray): AShape = self.fctShape(x,vShape)
        else: AShape = 0.0

        xi_in = self.pk2correlation(h=self.h,nk=self.nk,k=self.k,pk_in=pk_in,use_h_norm=use_h_norm) 
        #figure(10,)
        #plot(x,xi_in(x)*x*x)
        #ylabel('$r^2\\xi(r) [h^{-2}\mathrm{Mpc}^2]$')
        #xlabel('$r [h^{-1}\mathrm{Mpc}]$')
        return ( xi_in(x) + AShape )

    def params_plus_wavenumber(self):
        return self.h,self.Omega_m,self.redshift,self.nk,self.k

    def nr(self,xin,addx=100.,bias=np.nan,sigp=None,kaiser=False,damping=False,galaxy=False,vShape=None,returnInterpol=False,use_h_norm=True):
        #x=linspace(0,max(xin),1000)
        x = 10**np.linspace(np.log10(0.1),np.log10(np.max(xin+addx)),1000) #!#!#!#
        y=self.xi(x,bias=bias,sigp=sigp,kaiser=kaiser,damping=damping,galaxy=galaxy,vShape=vShape,use_h_norm=use_h_norm)*x**2
        theint=zeros(x.size)
        theint[1:]=1+3*integrate.cumtrapz(y,x=x)/x[1:]**3 
        ff=interpolate.interp1d(x,theint,bounds_error=False)
        if returnInterpol==True: return(ff)
        else:                    return(ff(xin))

    def rh_nr(self,threshold=1.01,bias=np.nan,sigp=None,kaiser=False,damping=False,galaxy=False,vShape=None,use_h_norm=True):
        x=np.linspace(10,1000,100)
        ff=interpolate.interp1d(self.nr(x,bias=bias,sigp=sigp,kaiser=kaiser,damping=damping,galaxy=galaxy,vShape=vShape,use_h_norm=use_h_norm)[::-1],x[::-1],bounds_error=False)
        return(ff(threshold))

    def d2(self,xin,min=0.1,addx=100.,bias=np.nan,sigp=None,kaiser=False,damping=False,galaxy=False,vShape=None,returnInterpol=False,use_h_norm=True):
        x=10**np.linspace(np.log10(min),np.log10(np.max(xin+addx)),1000)
        lognr=np.log10(self.nr(x,bias=bias,sigp=sigp,kaiser=kaiser,damping=damping,galaxy=galaxy,vShape=vShape,use_h_norm=use_h_norm))
        #dlogx=logx[1]-logx[0]
        dlogx = np.gradient(np.log10(x))
        thed2=np.gradient(lognr)/dlogx+3.
        ff=interpolate.interp1d(x,thed2,bounds_error=False)
        #figure(11,)
        #plot(x,ff(x))
        #ylabel('$\mathcal{D}_2(r)$')
        #xlabel('$r [h^{-1}\mathrm{Mpc}]$')
        if returnInterpol==True: return(ff)
        else:                    return(ff(xin))
        
    def rh_d2(self,threshold=2.97,bias=np.nan,sigp=None,kaiser=False,damping=False,galaxy=False,vShape=None,use_h_norm=True):
        x=np.linspace(10,1000,1000)
        ff=interpolate.interp1d(self.d2(x,bias=bias,sigp=sigp,kaiser=kaiser,damping=damping,galaxy=galaxy,vShape=vShape,use_h_norm=use_h_norm),x,bounds_error=False)
        return(ff(threshold))

    # Modelling Non-Linearities #############################
    def dambing_factor_lorentz(self,h=0.7,fgrowth=0.85,bias=2.,sigp=300.,k_theory=None):

        beta = fgrowth/bias
        HH = 100*h**2 # km/s/Mpc

        A_num = 2*HH*k_theory*beta*sigp
        B_num = -6.*(HH**2)*beta + (k_theory**2)*(6.+beta)*sigp**2
        C_num = 3.*np.sqrt(2.)*(-2.*HH*HH*beta+(k_theory**2)*sigp**2)**2
        D_num = np.arctan(k_theory*sigp/(np.sqrt(2.)*HH) )
        E_den = 3*(k_theory**5.) * (sigp**5.)

        result = HH*(A_num*B_num + C_num*D_num)/E_den 

        return result

    def dambing_factor_gaussian(self,h=0.7,fgrowth=0.85,bias=2.,sigp=300.,k_theory=None):
        """ taken from Mathematica Calculation """

        beta = fgrowth/bias
        HH = 100*h**2 # km/s/Mpc

        if type(sigp)==np.ndarray:
            sigpV2 = (sigp[1]**2 - sigp[0]**2)
            A_num = np.exp(-(k_theory**2.*sigp[1]**2.)/(2.*HH**2.)  )
            B_num = ( np.sqrt(2)*k_theory*beta*np.sqrt(sigpV2)*( 3*HH**2*beta + k_theory**2*(2+beta)*sigpV2) )/HH**3
            C_num = np.exp(k_theory**2*sigpV2/2/HH**2)*np.sqrt(np.pi)*special.erf( k_theory*np.sqrt(sigpV2/2)/HH )
            D_num = 3*beta**2  + 2*k_theory**2*beta*sigpV2/HH**2 + k_theory**4*sigpV2**2/HH**4
            E_den = np.sqrt(2.)*(k_theory**5.) * sigpV2**(5./2.)

            result = HH**5*A_num*( -B_num + C_num*D_num     )/E_den             
        else: 
            A_num = np.exp(-(k_theory**2.*sigp**2.)/(2.*HH**2.)  )
            B_num = - 6. * HH**3. * k_theory * beta**2. * sigp  -  2. * HH * k_theory**3. * beta * (2.+beta) * sigp**3.
            C_num = 3.* HH**4. * beta**2.  +  2*HH**2 * k_theory**2. * beta * sigp**2. + k_theory**4. * sigp**4.
            D_num = special.erf(k_theory*sigp/(np.sqrt(2.)*HH) )*np.sqrt(2.*np.pi)
            E_den = 2.*(k_theory**5.) * (sigp**5.)

            result = HH*(A_num*B_num + C_num*D_num)/E_den 

        return result

    def dambing_factor_gaussian_perp(self,h=0.7,fgrowth=0.85,bias=2.,sigp=None,k_theory=None):
        """ taken from Mathematica Mathematica Calculation"""
        beta = fgrowth/bias
        HH = 100*h**2 # km/s/Mpc

        if type(sigp)==np.ndarray:
            sigpV2= (sigp[1]**2 - sigp[0]**2)
            A_num = np.exp( -k_theory**2*(sigp[1]**2+3*sigp[0]**2)/8./HH**2 )
            B_num = k_theory*beta*np.sqrt(sigpV2)*(12.*HH**2*beta + k_theory**2*(8+beta)*sigpV2)/2./np.sqrt(2)/HH**3
            C_num = 2*np.exp( (k_theory/HH)**2*sigpV2/8. )*np.sqrt(np.pi)*special.erf(k_theory*np.sqrt(sigpV2)/2/np.sqrt(2.)/HH)
            D_num = 3*beta**2  + 2*k_theory**2*beta*sigpV2/HH**2 + k_theory**4*sigpV2**2/HH**4
            E_den = 2.*np.sqrt(2.)*(k_theory**5.) * sigpV2**(5./2.)

            result = HH**5*A_num*(-B_num+C_num*D_num)/E_den
    
        else: 
            expo0 = (k_theory**2*sigp**2)/HH**2/8
            A_num = np.exp(-expo0)
            B_num = -np.sqrt(2)*HH*k_theory*beta*sigp
            C_num = 12*HH**2*beta + k_theory**2*(8+beta)*sigp**2
            D_num = 8*np.exp(expo0)*np.sqrt(np.pi)*special.erf(k_theory*sigp/2/np.sqrt(2)/HH)
            E_num = 3*HH**4*beta**2 + 2*HH**2*k_theory**2*beta*sigp**2 + k_theory**4*sigp**4
            F_den = 8*np.sqrt(2)*(k_theory**5)*(sigp**5)

            result = HH*A_num*( B_num*C_num + D_num*E_num )/F_den

        return result

    def dambing_factor_gaussian_para(self,h=0.7,fgrowth=0.85,bias=2.,sigp=None,k_theory=None):
        """ taken from Mathematica Calculation"""
        beta = fgrowth/bias
        HH = 100*h**2 # km/s/Mpc
        if type(sigp)==np.ndarray:
            sigpV2= (sigp[1]**2 - sigp[0]**2)

            expo0 = (k_theory*sigp[0]/HH)**2
            expo1 = (k_theory*sigp[1]/HH)**2
            expo2 = (k_theory**2.)*sigpV2/HH**2

            exp0 = np.exp(  - 0.5*expo0 )
            exp1 = np.exp(  - 0.5*expo1 )
            exp2 = np.exp( (3./8)*expo2 )
            exp3 = np.exp(    0.5*expo2 )
            exp4 = np.exp( -((k_theory/HH)**2)*(4*sigp[1]**2 + 3.*sigp[0]**2)/8. )
            exp5 = np.exp( (3./8)*expo0 )
            exp6 = np.exp( (3./8)*expo1 )
            exp7 = np.exp(  (k_theory/HH)**2*(4*sigp[1]**2 - sigp[0]**2)/8. )
            
            A_num = 4.*exp0*k_theory**4.*np.sqrt(2.*np.pi)*sigpV2**2
            spErf0 = special.erf( k_theory*np.sqrt(sigpV2)/np.sqrt(2.)/HH/2. )
            spErf1 = special.erf( k_theory*np.sqrt(sigpV2)/np.sqrt(2.)/HH )
            B_num =  spErf1 - spErf0
            C_num = 8.*exp1*HH*(k_theory**2)*beta*sigpV2
            D_num = (exp2 - 2.)*k_theory*np.sqrt(sigpV2) - exp3*HH*np.sqrt(2.*np.pi)*(spErf0 - spErf1)
            E_num = exp4*HH*beta**2
            F_num =-k_theory*np.sqrt(sigpV2)*( -8.*exp5*(3*HH**2 +k_theory**2*sigpV2) + exp6*(12.*(HH**2) + (k_theory**2.)*sigpV2))
            G_num = 12.*exp7*HH**3*np.sqrt(2.*np.pi)*(spErf0 - spErf1)
            H_den = 8.*(k_theory**5.) * sigpV2**(5./2.)

            result = HH*( A_num*B_num + C_num*D_num - E_num*(F_num + G_num) )/H_den
        else:            
            expo0 = (k_theory*sigp)**2/HH**2
            expo1 =  expo0/2.
            expo2 =  3.*expo0/8.

            A_num = np.exp(-expo1) 
            B_num = HH*k_theory*beta*sigp
            C_num = 12.*( np.exp(expo2)-2.  )*(HH**2)*beta
            D_num = (k_theory*sigp)**2
            E_num = (8+beta)*np.exp(expo2) - 8.*(2.+beta)
            F_num = 4*np.exp(expo1)*np.sqrt(2*np.pi) 
            G_num = 3*(HH**2*beta)**2 + 2*(HH*k_theory*sigp)**2*beta + (k_theory*sigp)**4
            H_num = special.erf(k_theory*sigp/2/np.sqrt(2)/HH)
            I_num = special.erf(k_theory*sigp/np.sqrt(2)/HH)

            J_den = ( 8*(k_theory**5)*(sigp**5) )

            result = HH*A_num*( B_num*( C_num + D_num*E_num )  - F_num*G_num*(H_num-I_num) )/J_den

        return result

    ############# TEST FUNCTION for integral over mu ########
    def test_NL_factor(self,mu,k_theory,bias,sigp,fgrowth,h,kaiser=True,damping=False):
        """ Old function Not Used """
        beta = fgrowth/bias 
        HH   = 100.*h**2
        dambing_factor = 1.

        result = 1.0+0.0*mu        
        if kaiser or damping:
            if damping:
                result = np.exp(- .5*( k_theory*sigp*mu/HH )**2.)
                print('only damping' )
            if kaiser:
                print('only kaiser')
                if(damping):
                    if type(sigp)==np.ndarray :  dambing_factor = np.exp(- .5*( k_theory/HH )**2.*( (sigp[1]*mu)**2 - (1.-mu**2)*sigp[0]**2) )
                    else: dambing_factor = np.exp(- .5*( k_theory*sigp*mu/HH )**2.) 
                    print('+ damping')
                result = ( ( 1. + beta*mu**2. )**2. ) * dambing_factor

        return result

    def test_dambing_factor_gaussian(self,h=0.7,fgrowth=0.85,bias=2.,sigp=None,k_theory=None,muMin=0.0,muMax=1.0,kaiser=True,damping=False):

        Factor = np.zeros((self.nk))
        mu = np.linspace(muMin,muMax,1000)
        for k_index in np.arange(k_theory.size):
            Factor[k_index] = integrate.trapz(self.test_NL_factor(mu,k_theory[k_index],bias,sigp,fgrowth,h,kaiser=kaiser,damping=damping),x=mu)
        print('sth')
        if damping==False: print(Factor)
        
        return Factor
    ############# END: TEST FUNCTION for integral over mu ########

    def NL_factor(self,h=0.7,Omega0_m=0.3,z=0.,bias=2.,sigp=300.,nk=5000,k_theory=None,kaiser=True,damping=False):
        #if(dambing): print 'kaiser with dambing'
        fgrowth = self.fgrowthFUNC(z)
        #print fgrowth/bias
        HH = 100*h**2

        #!# SOS
        if damping==False:
            #Factor = 1.+ ((2./3.)*fgrowth/bias) + ((1./5.)*(fgrowth/bias)**2)
            Factor = Kaiser_TERM(fgrowth,bias)
        elif damping==True:
            #Factor = dambing_factor_lorentz(h=h,fgrowth=fgrowth,bias=bias,sigp=sigp,k_theory=k_theory) # KEEP
            Factor = self.dambing_factor_gaussian(h=h,fgrowth=fgrowth,bias=bias,sigp=sigp,k_theory=k_theory)
        elif damping=='perp':
            Factor = self.dambing_factor_gaussian_perp(h=h,fgrowth=fgrowth,bias=bias,sigp=sigp,k_theory=k_theory)
        elif damping=='para':
            Factor = self.dambing_factor_gaussian_para(h=h,fgrowth=fgrowth,bias=bias,sigp=sigp,k_theory=k_theory)
            ##### For testing formula taken from Mathematica #####
        elif damping=='test_perp':
            Factor = self.test_dambing_factor_gaussian(h=h,fgrowth=fgrowth,bias=bias,sigp=sigp,k_theory=k_theory,muMin=0.0,muMax=0.5,damping=True)
        elif damping=='test_para':
            Factor = self.test_dambing_factor_gaussian(h=h,fgrowth=fgrowth,bias=bias,sigp=sigp,k_theory=k_theory,muMin=0.5,muMax=1.0,damping=True)
        elif damping=='test_damp':
            Factor = self.test_dambing_factor_gaussian(h=h,fgrowth=fgrowth,bias=bias,sigp=sigp,k_theory=k_theory,muMin=0.0,muMax=1.0,damping=True) #!# SOS

        return Factor

    def fgrowthFUNC(self,z):
        return self.fgrowth(z) 

    # Non-Linearities #############################

    def fgrowth(self,z,gammaexp=0.55,if_paper='Coho2019'):
        ###return (Om*(1+z)**3.)**0.55
        # Parameterized Beyond-Einstein Growth
        # https://arxiv.org/pdf/astro-ph/0701317.pdf
        # after equation 3 second to last sentence of the paragraph
        # print( ( self.Omega_m*(1.+z)**3. /( self.Omega_m*(1.+z)**3. + 1. - self.Omega_m ) )**gammaexp )
        #print('Omega_k,Omega_m,Omega_L')
        #print(self.Omega_k,self.Omega_m,self.Omega_L)
        #print(1-(self.Omega_m+self.Omega_L)) 
        if if_paper=='Coho2019':
            res = ( self.Omega_m*(1.+z)**3. / cosmology.EE(z,self.Omega_m,self.Omega_L,omegaRad=0.0) )**gammaexp
        elif if_paper=='fnl2019':
            res = ( self.Omega_m*(1.+z)**3. / cosmology.EE(z,self.Omega_m,self.Omega_L,omegaRad=0.0)**2. )**gammaexp
        #print('fgrowth=%0.3f'%(res))
        return res

def Kaiser_TERM(fgrowth,bias):
    return 1.+ ((2./3.)*fgrowth/bias) + ((1./5.)*(fgrowth/bias)**2)

def cosmo_flagship(giveErrs=False):
    ''' Define parameters for CLASS soft '''
    ## Model Class: 
    # planck 2015, does not give sigma8
    params = {
        'h': 0.67,
        'omega_b': 0.049*0.67**2,
        'omega_cdm': (0.319-0.049)*0.67**2,
        'n_s': 0.96,
        'ln10^{10}A_s': 3.0494251, #np.log(2.125e-9*1e10), 
        #'A_s': np.exp(3.094)*1e-10,
        'Omega_k': 0.0,
        }
    sigma_8   = 0.83
    return params

def cosmo_Planck(giveErrs=False):
    ''' Define parameters for CLASS soft '''
    ## Model Class: 
    #planck 2015, does not give sigma8
    params = {
        'h': 0.6727,
        'omega_b': 0.02225,
        'omega_cdm': 0.1198,
        'n_s': 0.9645,
        'ln10^{10}A_s': 3.094, 
        #'A_s': np.exp(3.094)*1e-10,
        'Omega_k': 0.0,
        }
    sigma_8   = 0.831

    # omega_b/omega_cdm = 0.1857 = 1/5

    dparams={
        'h': 0.0066,
        'omega_b': 0.00016,
        'omega_cdm': 0.0015,
        'n_s': 0.0049,
        'ln10^{10}A_s': 0.034,
        #'Omega_k':0.0054,
        }

    if giveErrs: return params,dparams 
    else:        return params

def cosmo_Planck_Big(giveErrs=False):
    ''' Defined parameters in for Cosmology Use'''
    paramsPL = cosmo_Planck(giveErrs=False)
    paramsPL_out = {
    'h':paramsPL['h'],
    'Omega_m':(paramsPL['omega_b']+paramsPL['omega_cdm'])/paramsPL['h']**2,
    'Omega_L':1-(paramsPL['omega_b']+paramsPL['omega_cdm'])/paramsPL['h']**2 - paramsPL['Omega_k'], 
    'Omega_b':paramsPL['omega_b']/paramsPL['h']**2,
    'sigma8':0.831, 
    'n_s':paramsPL['n_s'],
    'Omega_k':paramsPL['Omega_k']
    }
    return paramsPL_out

def giveCosmoNames(giveExtra=False):
    ''' Define parameters for CLASS soft '''
    list1 = ['h', 'omega_b', 'omega_cdm' , 'n_s', 'ln10^{10}A_s']  
    if giveExtra: list1 +=['Omega_k','Omega_Lambda','w0_fld','wa_fld'] 
    return list1

def cosmo_PlanckAll(giveErrs=False,giveExtra=False):
    ''' Define parameters for CLASS soft '''
    cosmo = cosmo_Planck(giveErrs=giveErrs)
    
    params_extra = {
    'Omega_k':-0.004,
    'Omega_Lambda': 0.658,
    'w0_fld':-1.006,
    'wa_fld':1e-4,
    }

    dparams_extra={
        'Omega_k':0.012,
        'Omega_Lambda':0.013,
        'w0_fld': 0.045,
        'wa_fld':0.0054,
        }

    if giveExtra:
        if giveErrs: 
            cosmo[0].update(params_extra)
            cosmo[1].update(dparams_extra)
            return cosmo[0],cosmo[1]
        else:       
            cosmo.update(params_extra)
            return cosmo
    else:
        return cosmo

def cosmo_QPM_Big():
    ''' Defined parameters in for Cosmology Use'''
    #Cpars = [0.274,0.726,-1,0.8,0.7,0.046,0.95] # QPM Values
    parsQPM = {
                'h':0.7,
                'Omega_m':0.274,
                'Omega_L':0.726, 
                'Omega_b':0.046,
                'sigma8':0.8, 
                'n_s':0.95,
                'Omega_k':0.0
    }
    return parsQPM

def cosmo_QPM():
    ''' Define parameters for CLASS soft '''
    ## Model Class: 
    # QPM cosmology
    pQPM=cosmo_QPM_Big()
    params = {
                'h':          pQPM['h'],
                'omega_b':    pQPM['Omega_b']*pQPM['h']**2 ,
                'omega_cdm': (pQPM['Omega_m']-pQPM['Omega_b'])*pQPM['h']**2,
                'ln10^{10}A_s':np.log(2.1683570910090001e-9*1e10), #2.3e-9, 
                'n_s':pQPM['n_s'],
                'Omega_k': 0.0
                }
    sigma8QPM=pQPM['sigma8']

    return params

def cosmo_PL2013_Big():

    params_PL2013 = {
        'h': 0.6704,
        'n_s': 0.9619,
        'Omega_b': 0.022032/0.6704**2,
        'Omega_m': 0.3183,
        'Omega_L':1-0.3183-0.0,
        'sigma8':0.8347,
        'Omega_k': 0.0
        }  
    return params_PL2013

def cosmo_PL2013():
    ''' Define parameters for CLASS soft '''
    sigma8=0.8347
    params_PL2013_in  = cosmo_PL2013_Big()
    params_PL2013_out = {
        'ln10^{10}A_s': np.log(2.1683570910090001e-9*1e10), #2.3e-9,
        'n_s': params_PL2013_in['n_s'],
        'omega_b': params_PL2013_in['Omega_b']*params_PL2013_in['h']**2,
        'omega_cdm': (params_PL2013_in['Omega_m']-params_PL2013_in['Omega_b'])*params_PL2013_in['h']**2 ,
        'h': params_PL2013_in['h'],
        'Omega_k': params_PL2013_in['Omega_k'],
        }

    return params_PL2013_out

def QPM2PL_a():
    ''' ratios of DV(O_PL) / DV(O_QPM) for 5 zbins '''
    #print 'Not automatic: QPM2PL_a need to compute correctely'
    return np.array([ 1.02284334,  1.02106576,  1.01937222,  1.01776272,  1.01623618])

def cosmo_WZ_Big():
    ''' Defined parameters in for Cosmology Use'''
    parsWZ = {
    'h':0.71,
    'Omega_m':0.27,
    'Omega_L':0.73, 
    'Omega_b':0.04482,
    'sigma8':0.8, 
    'n_s':0.96,
    'Omega_k':0.0
    }
    return parsWZ

def cosmo_WZ():
    ''' Define parameters for CLASS soft '''
    pWZ=cosmo_WZ_Big()
    pars_CWZ = {
    'h':pWZ['h'],
    'omega_b': pWZ['Omega_b']*pWZ['h']**2 ,
    'omega_cdm': (pWZ['Omega_m']-pWZ['Omega_b'])*pWZ['h']**2,
    'ln10^{10}A_s':3.094, 
    'n_s':pWZ['n_s'],
    'Omega_k':pWZ['Omega_k']
    }
    sigma8=pWZ['sigma8']

    return pars_CWZ


################### VOIDING ###########################

"""
Eq 31 and nearby in 1409.3849 
"""

def eta_fnct(D=35.,number_of_voids=122907.,volume_in_Mpc_h_power_3=2000**3.):
    """ packing factor """
    mean_number_density = number_of_voids/volume_in_Mpc_h_power_3
    print(pi*mean_number_density*D**3./6.)
    return pi*mean_number_density*D**3./6.

def alpha_1_fnct(D=35.,number_of_voids=122907.,volume_in_Mpc_h_power_3=2000**3.):
    eta_input = eta_fnct(D=D,number_of_voids=number_of_voids,volume_in_Mpc_h_power_3=volume_in_Mpc_h_power_3)
    return ((1.+2.*eta_input)**2.)/((1.-eta_input)**4.)

def alpha_2_fnct(D=35.,number_of_voids=122907.,volume_in_Mpc_h_power_3=2000**3.):
    eta_input = eta_fnct(D=D,number_of_voids=number_of_voids,volume_in_Mpc_h_power_3=volume_in_Mpc_h_power_3)
    return -1.0*((1.+eta_input/2.)**2.)/((1.-eta_input)**4.)

def q_fnt(k,D=35.): return k*D

def c_fnct(k,D=35.,number_of_voids=122907.,volume_in_Mpc_h_power_3=2000**3.):
    q_in       = q_fnt(k,D=D)
    eta_in     = eta_fnct(    D=D,number_of_voids=number_of_voids,volume_in_Mpc_h_power_3=volume_in_Mpc_h_power_3)
    alpha_2_in = alpha_2_fnct(D=D,number_of_voids=number_of_voids,volume_in_Mpc_h_power_3=volume_in_Mpc_h_power_3)
    alpha_1_in = alpha_1_fnct(D=D,number_of_voids=number_of_voids,volume_in_Mpc_h_power_3=volume_in_Mpc_h_power_3)
    Factor_1   = -D**3./(2.*pi**2. *q_in**3. )
    Term_1 = alpha_1_in*( sin(q_in) - q_in*cos(q_in) ) 
    Term_2 = (6.*eta_in*alpha_2_in/q_in)*( 2*q_in*sin(q_in) + (2. - q_in**2.)*cos(q_in) - 2.  )
    Term_3 = (eta_in*alpha_1_in/2./q_in**3.)*( 4.*q_in*(q_in**2.-6.)*sin(q_in) - (24.-12.*q_in**2. + q_in**4. )*cos(q_in) + 24. )
    c_out = Factor_1*( Term_1 + Term_2 + Term_3 )
    return c_out

def P_exclusion_fnt(k,D=35.,number_of_voids=12207.,volume_in_Mpc_h_power_3=2000**3.,if_Gaussian_Damping=False,sig_G=10.,kstar=0.2):
    mean_number_density = number_of_voids/volume_in_Mpc_h_power_3
    c_in = c_fnct(k,D=D,number_of_voids=number_of_voids,volume_in_Mpc_h_power_3=volume_in_Mpc_h_power_3)
    result = c_in / (1.-(2.*pi)**3.*mean_number_density*c_in ) 
    if if_Gaussian_Damping:
        result=result*exp(-0.5*(sig_G*(k-kstar) )**2.)         
    return result

################### End: VOIDING ###########################


def Conv2labels(stringi):
    
    if   stringi=='n_s':          return '$%s$'%stringi
    elif stringi=='w0_fld':       return '$w_0$'
    elif stringi=='wa_fld':       return '$w_a$'
    elif stringi=='h':            return '$h$'
    elif stringi=='Omega_Lambda': return '$\Omega_{\Lambda}$'
    elif stringi=='omega_cdm':    return '$\omega_{cdm}$'
    elif stringi=='Omega_m':      return '$\Omega_{m}$'
    elif stringi=='b0':           return '$b_{0}$'
    else:                         return '$\%s$'%stringi
