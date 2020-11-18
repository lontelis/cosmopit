import numpy as np
from numpy import *
from pylab import *
import warnings
from scipy import integrate
from scipy import interpolate
print('# Authors: P.Ntelis')
print('# original code taken from J.C.Hamilton in June 2014')
c=3e8       #m.s-1
H0_over_h= 1000*100 #m.s-1.Mpc-1s
dHubble = c/H0_over_h # Mpc/h

def get_dh():
    return dHubble

def H0_def(h):
    """ return H0 in m/s/Mpc"""
    return H0_over_h*h

def properdistance(z,omegam=0.3,omegax=0.7,w0=-1,w1=0,wz=None,omegaRad=0.00,compute_which_chi='comoving_distance'):
    """
        Gives the proper distance in the defined cosmology
        The c/H0 factor is ommited
        Returns dist(z), w(z), omegax(z), H(z), curvature
    """
    # if no wz on input the calculate it from w0 and w1
    if wz is None: wz=w0+(z*1./(1.+z))*w1
    # calculate evolution of omegax accounting for its equation of state updated to just simple formulat, instead of an integral
    """
    omegaxz=zeros(z.size)
    omegaxz[0]=omegax
    omegaxz[1:z.size]=omegax*exp(3*integrate.cumtrapz((1.+wz)/(1.+z),x=z))
    """
    omegaxz=omegax*(1+z)**(3.*(1+w0+w1))*np.exp(-3.*w1*z/(1+z))

    # figure(4),plot(z,omegaxz),show()
    # curvature
    omega=omegam+omegax+omegaRad
    if omega>1: curv=1
    if omega<1: curv=-1
    if omega==1: curv=0
    kk=abs(1.-omega)
    #print omegam,kk
    # calculation of E(z)=H(z)/H_0

    Ez=sqrt( (1.-omega)*(1+z)**2+omegaxz+omegam*(1+z)**3 + omegaRad*(1+z)**4 )

    # calculate chi
    chi=zeros(z.size)
    if compute_which_chi=='comoving_distance':
        chi[1:z.size]=integrate.cumtrapz(1./Ez,x=z)
    elif compute_which_chi=='to_compute_the_age_of_the_universe':
        chi[1:z.size]=integrate.cumtrapz(1./Ez/(1.+z),x=z)

    #calculate proper distance
    if curv==1:  dist=sin(sqrt(kk)*chi)/sqrt(kk)
    if curv==-1: dist=sinh(sqrt(kk)*chi)/sqrt(kk)
    if curv==0:  dist=chi
    
    return(dist,wz,omegaxz,Ez,curv)

def get_dist(z,type='proper',omegaRad=0.0,params=[0.3,0.7,-1,0],wz=None,z2radial=False):
    """
    Returns distances in Mpc/h in the defined cosmology
    type can be :
       prop : proper distance
       dl   : Luminosity distance
       dang : Angular distance
       dangco : Comoving angular distance
       wz : equation of state as a function of z
       omegaxz : omegax(z)
       hz : h(z)
       curv : curvature
       vco : Comoving volume
       rapp : proper*H(z)
    """

    omegam=params[0]
    omegax=params[1]
    w0=params[2]
    w1=params[3]
    
    if isinstance(z, np.ndarray): zmax = z.max()
    else:                         zmax = z

    zvalues=linspace(0.,zmax*1.5,1e5)

    dist,wz,omegaxz,Ez,curv=properdistance(zvalues,omegam=omegam,omegax=omegax,w0=w0,w1=w1,wz=wz,omegaRad=omegaRad)

    if type=='proper':
        res=dist*dHubble
    elif type=='dl':
        res=dist*(1+zvalues)*dHubble
    elif type=='dang':
        res=dist/(1+zvalues)*dHubble
    elif type=='dangco':
        res=dist*dHubble
    elif type=='wz':
        res=wz
    elif type=='omegaxz':
        res=omegaxz
    elif type=='hz':
        res=Ez*H0_over_h
    elif type=='curv':
        res=curv
    elif type=='vco':
        res=dist**2/Ez*(dHubble)**3 #/H0_over_h # in h-3.Mpc3
    elif type=='rapp':
        res=dist*Ez*c #??
    else:
        print("This type does not exist:",type)
        res=-1
    if z2radial: 
        f=interpolate.interp1d(res,zvalues)
    else: 
        f=interpolate.interp1d(zvalues,res) 
    return(f(z))

def VolCalcSurvey(zCentral,DeltaZrange,OmegaSky=1e4,params=[0.3,0.7,-1,0],wz=None):
    ''' Returns Comoving Volume in (Mpc/h)**3 of a survey for any cosmology with:
        OmegaSky # in deg**2
        zcentral 
        DeltaZrange '''
    Vcomo = get_dist(zCentral,type='vco',params=params,wz=wz)
    Volume= Vcomo*DeltaZrange*OmegaSky*(pi/180.)**2
    return Volume # in (Mpc/h)**3

def VolSurvey(zCentral,DeltaZrange,OmegaSky=1e4,params=[0.3,0.7,-1,0],wz=None):
    ''' 
        Moreaccurate for flat LCDM cosmologies
        Returns Comoving Volume in (Mpc/h)**3 of a survey with input:
        OmegaSky # Angular surface in deg**2
        zcentral # central redshift
        DeltaZrange # total redshift bin
    '''
    Volume = OmegaSky*(pi/180.)**2.*( get_dist(zCentral+DeltaZrange/2.,params=params,wz=wz)**3. - get_dist(zCentral-DeltaZrange/2.,params=params,wz=wz)**3.)/3.
    return Volume # in (Mpc/h)**3

def D_V(z,h=0.7,omegam=0.3,omegax=0.7,w0=-1,w1=0,wz=None,NNz=1000):
    ''' D_V analytical in Mpc '''
    if isinstance(z, np.ndarray): zmax = z.max()
    else:                         zmax = z
    zvalues=linspace(0.,zmax*1.0,NNz)
    dist,wz,omegaxz,Ez,curv=properdistance(zvalues,omegam=omegam,omegax=omegax,w0=w0,w1=w1,wz=wz)
    result = (dHubble/h) * ( (z/Ez ) * dist**2. )**(1./3.)
    return result[-1:]

def EE(z,omegam,omegax,omegaRad=0.0,w0=-1.0,wa=0.0):
    # gives the E(z)=H(z)/H0
    omegaxz= omegax*(1+z)**(3.*(1+w0+wa))*np.exp(-3.*wa*z/(1+z))
    omega  = omegaxz+omegam+omegaRad
    res    = sqrt( (1.-omega)*(1+z)**2+omegaxz+omegam*(1+z)**3 + omegaRad*(1+z)**4 )
    return(res)

def integrant(z,omegam,omegax):
    return (1+z)/(EE(z,omegam,omegax))**3 

def D1(z,params=[0.3,0.7,-1,0]):
    """ 
    Returns the Growthfactor given by 
    -
    equation linear growth factor
    Correction on formula:
        since we have the normalization
        we can put E(z) instead H(z)
        So no need of h!!!!
    """
    omegam=params[0]
    omegax=params[1]
    w0=params[2]
    w1=params[3]

    D1_z = z*0.
    for i in np.arange(np.size(z)):
        D1_z[i] = EE(z[i],omegam,omegax)*integrate.quad(integrant, z[i], np.infty, args=(omegam,omegax))[0]

    Normalization = integrate.quad(integrant, 0., np.infty, args=(omegam,omegax))[0] # ensures D1(z=0)=1

    result = D1_z/Normalization
    print('Normalization=',Normalization)
    #print('result=',result)
    return(result)

def omega_nuf(sum_mass_nu=0.06,Neff=3.046):
    result = 0.0107*sum_mass_nu/1.0
    return(result)

def r_d(omega_cdm=0.1198,omega_b=0.02225,nonRelativistic=True,sum_mass_nu=0.06,Neff=3.046):
    """
        https://arxiv.org/pdf/1411.1074.pdf
        p5, eq 16
        Calibrated drag epoch 
        numerical formula
    """
    omega_cb=omega_cdm+omega_b
    omega_nu = omega_nuf(sum_mass_nu=sum_mass_nu,Neff=Neff)
    if nonRelativistic:
        result = 55.154 * exp( -72.3*(omega_nu+0.0006)**2. )  /( (omega_cb**0.25351)*(omega_b**0.128070) )
    else:
        result = 56.067 * exp( -49.7*(omega_nu+0.0020)**2  )  /( (omega_cb**0.24360)*(omega_b**0.128876) ) / ( 1+(Neff-3.046)/30.60 )
    return result 

def z_d_EH1998(Omegam0=0.313,Omegab0=0.44,h=0.6727):
    """ redshift to the drag epoch
    eq. 4 from https://arxiv.org/pdf/astro-ph/9709112.pdf
    \Omega_0 \simeq 1 is the total density ratio in an Einstein-de-Sitter Universe. 
    """
    Omegam0h2 = Omegam0*h**2.
    Omegab0h2 = Omegab0*h**2.
    b_1 = 0.313* ( ( Omegam0h2 )**(-0.419) )*( 1+0.607*Omegam0h2**0.674 )
    b_2 = 0.238*Omegam0h2**0.233
    result = 1291.*( ( Omegam0h2**0.251  ) / ( 1+0.659*Omegam0h2**0.828 ) )*( 1+b_1*Omegab0h2**b_2 )
    return(result)

def c_s_EH1998_approximate(z,Omegab0=0.44,h=0.6727):
    """
    From https://arxiv.org/pdf/astro-ph/9709112.pdf
    eq. 5 and previous text
    c_s = 1/sqrt{2(1+R)}
    """
    Theta_27 = 2.728/2.7 #
    T_CMB = 2.7*Theta_27 # in K (Fixsen et al. 1996)
    R_EH1998_approximate = 31.5*Omegab0*h**2.*Theta_27**(-4.)*(1000./z)
    res = c/1e3/np.sqrt(3.*(1.+R_EH1998_approximate)) # in km/s
    return res

def c_s_EH1998(z,h=0.6727,Omegab0=0.44,w_b=0.0,Omegagamma0=0.001):
    """
    From https://arxiv.org/pdf/astro-ph/9709112.pdf
    eq. 5 expanded 
    R(z) \equiv (3\rho_b(z) )/(4\rho_{\gamma}(z) ) \rightarrow (3\Omega_b(z) )/ (4\Omega_{\gamma}(z) ) by deviding by \rho_c, 
    in the standard FLRW metric
    This is translated to the 
    case of a equation stated dependend quantity as:
    (3\Omega_{b,0}(1+z)^3  )/ (4\Omega_{\gamma,0}()^{4} )
    where w_b=-1./3. for the standard FLRW metric
    and previous text
    c_s = 1/sqrt{2(1+R(z))}
    """
    Theta_27 = 2.728/2.7 #
    T_CMB = 2.7*Theta_27 # in K (Fixsen et al. 1996)
    Nominator   = 3.*Omegab0*(1+z)**(3.*(w_b+1)) 
    Denominator = 4.*Omegagamma0*(1+z)**4.
    R_EH1998 = Nominator/Denominator
    res = c/1e3/np.sqrt(3.*(1.+R_EH1998)) # in km/s
    return(res)

def c_s_z_EH1998_div_by_Ez(z,h=0.6727,Omegam0=0.31,OmegaLambda0=1.-0.31,Omegab0=0.044,Omegagamma0=0.01):
    """
    From https://arxiv.org/pdf/astro-ph/9709112.pdf
    eq. 5 expanded 
    R(z) \equiv (3\rho_b(z) )/(4\rho_{\gamma}(z) ) \rightarrow (3\Omega_b(z) )/ (4\Omega_{\gamma}(z) ) by deviding by \rho_c, 
    in the standard FLRW metric
    This is translated to the 
    case of a equation stated dependend quantity as:
    (3\Omega_{b,0}(1+z)^3 )/ (4\Omega_{\gamma,0}()^{4} )
    and previous text
    c_s = 1/sqrt{2(1+R(z))}
    approximate formula
    """
    Theta_27 = 2.728/2.7 #
    T_CMB = 2.7*Theta_27 # in K (Fixsen et al. 1996)
    Nominator   = 3.*(Omegab0*(1+z)**3.)
    Denominator = 4.*Omegagamma0*(1+z)**4.
    R_EH1998 = Nominator/Denominator
    c_s_z_EH1998 = dHubble/np.sqrt(3.*(1.+R_EH1998)) # in Mpc/h
    res = c_s_z_EH1998/EE(z,omegam=Omegam0,omegax=OmegaLambda0,omegaRad=Omegagamma0)
    return(res)

def radius_of_sound_horizon_given_by_c_s_z_EH1998_div_by_Ez(z_d_input,h=0.6727,Omegam0=0.31,OmegaLambda0=1.-0.31,Omegab0=0.044,w_b=-1./3.,Omegagamma0=0.0001,epsabs=1.49e-07,epsrel=1.49e-07):
    """ https://arxiv.org/pdf/astro-ph/9709112.pdf """
    res = integrate.quad( c_s_z_EH1998_div_by_Ez , z_d_input, np.inf, args=(h,Omegam0,OmegaLambda0,Omegab0,w_b,Omegagamma0),epsabs=epsabs, epsrel=epsrel )[0]
    return(res)

def D_C_approx(z,params=[0.3,0.7,-1,0],wz=None):
    ''' D_C approximated in Mpc/h https://arxiv.org/pdf/1411.1074.pdf'''
    return get_dist(z,type='proper',params=params,wz=wz)

def D_M_approx(z,params=[0.3,0.7,-1,0],wz=None):
    ''' D_M approximated in Mpc/h https://arxiv.org/pdf/1411.1074.pdf'''
    return D_C_approx(z,params=params,wz=wz)*( 1+ (1/6.)*(1-params[1]-params[0])* (D_C_approx(z,params=params,wz=wz)/c*H0_over_h)**2 )

def D_H_approx(z,params=[0.3,0.7,-1,0],wz=None):
    ''' D_H approximated in Mpc/h https://arxiv.org/pdf/1411.1074.pdf'''
    return c/get_dist(z,type='hz',params=params,wz=wz)

def D_V_approx(z,params=[0.3,0.7,-1,0],wz=None):
    ''' D_V approximated in Mpc/h https://arxiv.org/pdf/1411.1074.pdf'''
    return ( z*D_H_approx(z,params=params,wz=wz) * D_M_approx(z,params=params,wz=wz)**2 )**(1./3.)

def z2age_of_univ(z,H0=67.27,omegam=0.3,omegax=0.7,w0=-1,w1=0,wz=None,omegaRad=0.00,in_timeunits='Gyr'):
    dist,wz,omegaxz,Ez,curv=properdistance(z,omegam=omegam,omegax=omegax,w0=w0,w1=w1,wz=wz,omegaRad=omegaRad,compute_which_chi='to_compute_the_age_of_the_universe') #'comoving_distance')# 'to_compute_the_age_of_the_universe')
    integral = dist
    convert_H0_to_time_units = 3.08*1e19/H0 # in seconds
    time_z   = integral*convert_H0_to_time_units
    print(curv)
    if in_timeunits=='sec':
        return time_z   
    elif in_timeunits=='yr':
        return time_z/(365.*24.*3600.)     #31556926 simeq 365.*24*3600.
    elif in_timeunits=='Gyr':
        return time_z/(365.*24.*3600.)/1e9 #31556926 simeq 365.*24*3600.

def dclock(z,AA=0.58,BB=.64):
    ''' the estimation of distance-redshift clock relation'''
    return dHubble * ( np.log(AA*z+BB)/AA - np.log(BB)/AA )

"""
# test light-travel distance or look-back time 
# with https://en.wikipedia.org/wiki/Distance_measures_(cosmology)
figure(1),clf()
z=10**(linspace(log10(1e-10),log10(10000.),100000))
plot(z,cosmology.z2age_of_univ(z,H0=72.0,omegam=0.268,omegax=0.732,omegaRad=0.266/3454,in_timeunits='Gyr'))
xlim(1e-4,1e4), ylim(1e-3,1e4),
xscale('log'),yscale('log'),grid()
xlabel('$z$',size=20),ylabel('$t(z)$ [Gyr]',size=20)
draw(),show()

figure(2),clf()
z=10**(linspace(log10(1e-10),log10(10000.),100000))
plot(z,cosmology.z2age_of_univ(z,H0=72.0,omegam=0.268,omegax=0.732,omegaRad=0.266/3454,in_timeunits='Gyr'))
xlim(2.0,11.0), ylim(10.0,20.0),
xscale('log'),yscale('linear'),grid()
xlabel('$z$',size=20),ylabel('$t(z)$ [Gyr]',size=20)
draw(),show()

# test different epochs with different redshifts
H0 = 67.27 #km/s/Mpc
omegam=0.3
omegax=0.7
z=10**(linspace(log10(1e-10),log10(10000.),100000))
#epoch from the initial fluctuations
age_of_the_universe=cosmology.z2age_of_univ(z,H0=H0,omegam=omegam,omegax=omegax,in_timeunits='Gyr')[100000-1]
time_from_the_initial_fluctuations=age_of_the_universe-cosmology.z2age_of_univ(z,H0=H0,omegam=omegam,omegax=omegax,in_timeunits='Gyr')

"""

# Test Hubble Enough
"""
from Cosmology import cosmology
dh = cosmology.get_dh()
z = linspace(0,10,1000)

clf()
Om=array([0.2,0.3,0.4])
#w0=array([-0.5,-1.,-1.5])
for i in range(len(Om)):
    #for j in range(len(w0)):
    
    params=[Om[i],1-Om[i],-1.,0]
    d_LCDM = cosmology.get_dist(z,params=params)
    
    subplot(311)
    plot(z,d_LCDM-z*dh,label='$\Omega_m,w_0$=%0.1f,%0.1f'%(Om[i],-1) )
    legend(loc=3,numpoints=1),
    ylim(-400,0),xlim(0,1),grid()
    ylabel('$d_{LCDM}-d_{H}\ [h^{-1}\mathrm{Mpc}]$',size=15)

    subplot(312)
    plot(z,d_LCDM,label='$\Omega_m,w_0$=%0.1f,%0.1f'%(Om[i],-1) )
    if i==len(Om)-1: plot(z,z*dh,label='Hubble' )
    ylabel('$d_{LCDM}\ [h^{-1}\mathrm{Mpc}]$',size=15),
    ylim(0.,4000),xlim(0,1),grid()
    legend(loc=2,numpoints=1)

    subplot(313)
    plot(z,d_LCDM/(z*dh),label='$\Omega_m,w_0$=%0.1f,%0.1f'%(Om[i],-1) )    
    ylabel('$d_{LCDM}/d_{H}\ [h^{-1}\mathrm{Mpc}]$',size=15),xlim(0,1),grid()
    xlabel('$z$',size=25)
    legend(loc=3,numpoints=1)


#plot(z,z*dh,'k--',label='Hubble'),


def integrant_clock(z,AA=-0.3,BB=2.0):
    return (1+z)*(AA*z+BB)

def dclock_old(z,AA=0.3,tF=0.03):
    ''' Complicated not use'''
    # tF according to A(Z,SFR) relation. d_clock(z; tF=0.03) ~ d_comov(z;[Om,Ol]=[0.3,0.7])
    timeFactor = tF/(1e9*365*24*3600.)
    #integral = integrate.cumtrapz(integrant_clock(z,AA=-0.3,BB=2.0),x=z)

    m2Mpc = 3.24077929e-23
    return c*m2Mpc*z*(z+2.)/2.*AA/timeFactor # dist in Mpc

"""

