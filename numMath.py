import numpy as np
from numpy import *
from scipy import linalg,sparse
import matplotlib.pyplot as plt
from matplotlib import cm

# Hellpfull numerics by P.Ntelis June 2014

def blockMat1row(matr):
    temp = []
    for i in range(len(matr)):
        temp.append(matr[i])
    return sparse.block_diag(temp).toarray()

def testArrBurn(arr,doplot=False,fmt='.',color='r',figN=3):
    kkk=0
    temp_m = zeros(arr.size-kkk)
    temp_s = zeros(arr.size-kkk)
    totalm = mean(arr)
    for i in arange(temp_m.size):
        temp_m[i] = mean(arr[:i+1])
        temp_s[i] = std(arr[:i+1]) #/kkk
    if doplot:
        plt.figure(figN),plt.clf()
        plt.subplot(311)
        plt.plot(arr,'b.')
        plt.plot(temp_m,color+'-',label='mean')
        plt.plot(temp_m+temp_s,color+'--',label='std')
        plt.legend(numpoints=1,frameon=False,loc=4)
        plt.subplot(312)
        plt.plot(temp_m/totalm,color+'-',label='mean')
        plt.legend(numpoints=1,frameon=False,loc=4)
        plt.plot(arange(temp_m.size),arange(temp_m.size)*0.+1.,'k--')
        plt.subplot(313)
        plt.plot(temp_s/totalm,color+'--',label='std/totalm')
        plt.legend(numpoints=1,frameon=False,loc=4)
        print totalm

    return temp_m,temp_s

def burnChains(chains,kmin=0):
    ''' Try to find the bug '''
    newChains=dict(chains) # dict(chains)
    kmax = newChains[newChains.keys()[0]].size
    for k in newChains.keys(): newChains[k] = newChains[k][kmin:kmax]
    return newChains

def chainsTo2Darray(chain_in,PermMatlist=None):
    ''' convert chain format to 2Darray '''
    chains = []
    for i in chain_in['chains'].item().keys(): chains.append(chain_in['chains'].item()[i])
    chains=array(chains)
    if type(PermMatlist)==list:
        return PermMat(chains,PermMatlist)
    else:
        return chains

def average_realisations(datasim):
    ''' 
        general stat of a chain array: 
        out: means 
             stds
             covariance matrix
             correlation matrix
    '''
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

def insertP(arrmat):
    ''' inserts an additional column, and row with zeros on a metrix'''
    a_zero_c = np.zeros(len(arrmat))
    a_zero_r = np.zeros(len(arrmat)+1)
    arrmat_insert_c = np.insert(arrmat,len(arrmat),a_zero_c,axis=1)
    arrmat_insert_r = np.insert(arrmat_insert_c,len(arrmat_insert_c),a_zero_r,axis=0)
    return arrmat_insert_r

def addVal_Mat(arrmat,val=0.0):
    ''' adds a value on the last diagonal element of a matrix '''
    arrmat[len(arrmat)-1,len(arrmat)-1] =+ val
    return arrmat

def addRCVal(arrmat,val=0.0):
    ''' 
        add additional column and row and 
        puts a value on the last diagonal 
        of a symmtric matrix
    '''
    new_mat = insertP(arrmat)
    return addVal_Mat(new_mat,val=val)
    

def PermMat(mat,permut):
    ''' Permut and Extract Raws and Columns of a Matrix '''
    return (mat[permut,:])[:,permut]

def AddColMat(mat,zeros=True):
    if zeros:
        newmat = np.zeros((mat.shape[0]+1,mat.shape[1]+1))
    else:
        newmat = np.ones((mat.shape[0]+1,mat.shape[1]+1))   
    newmat[:-1,:-1] = mat
    return newmat

def cut_data(r,Obs,covmat,wok):
    '''                                                                                                                                                                                  
        cuts data according to wok                                                                                                                                                       
        use: r_cut,Obs_cut,cov_cut = cut_data(r,Obs,covmat,wok)                                                                                                                          
    '''
    r_cut   = r[wok]
    Obs_cut = Obs[wok]
    cov_cut = (covmat[wok[0],:])[:,wok[0]]
    return r_cut,Obs_cut,cov_cut

def flattening_func(x,f,xlim=10,fval=None):
    """ flattens part of a function:
        f(x>xlim) = fval  
        or
        f(x>xlim) = mean(f(x>xlim))  """

    wok = np.where(x<xlim)
    wNok= np.where(x>xlim)

    if fval==None:
        fval = np.mean(f[wNok]) # take the average after xlim

    f_part= f[wNok]*0.0 + fval
    f_new = np.concatenate((f[wok],f_part)) 
    
    return f_new

def delta_kron(i,j):
    if i!=j: delta=0.0
    else: delta=1.0
    return delta
    
def open_cov(covmat,vector,kronecker=True):

    dims = np.shape(covmat)

    covmat_opened = covmat*0.0

    for i in np.arange(dims[0]):
        for j in np.arange(dims[1]):
            
            delta = 1.0
            if kronecker==True:
                delta=delta_kron(i,j)
            #print i,j,delta
            
            covmat_opened[i][j] = covmat[i][j] * (1. + vector[i]*vector[j]*delta )
    return covmat_opened

def vec2cov(vector):
    dims = np.shape(vector)
    covmat = np.zeros((dims[0],dims[0]))
    for i in np.arange(dims[0]):
        for j in np.arange(dims[0]):
            covmat[i][j]=vector[i]*vector[j]
    return covmat

def open_cov_Pierros(covmat,vector):
    ''' return covmat_opened = covmat * ( Ones + diag(covmat)*vector ) '''
    one_matrix = np.eye(np.shape(covmat)[0])
    correction = one_matrix*covmat*vector
    covmat_opened = covmat+correction
    return covmat_opened

def kth_diag_indices(matrix, k):
    '''how to take indices of matrix to select only kth-diagonals'''
    rows, cols = np.diag_indices_from(matrix)
    if k < 0:
        return rows[:k], cols[-k:]
    elif k > 0:
        return rows[k:], cols[:-k]
    else:
        return rows, cols

def cov_smooth(covmat):
    dim = np.shape(covmat)[0]

    mean_diag = np.zeros(dim)
    covmat_new = covmat*0.0 + covmat
    for i in xrange(dim):
        mean_diag[i]  = np.mean( np.diagonal(covmat,offset=i))

    for i in xrange(dim):
        if (i>0): # smooth all except the diagonal
            covmat_new[kth_diag_indices(covmat_new, i)] = mean_diag[None,i]
            covmat_new[kth_diag_indices(covmat_new, -i)] = mean_diag[None,i]

    return covmat_new

def give_msc(x,y,decimals=2):

    meanx,meany,stdx,stdy = np.around( np.mean(x), decimals=decimals) ,np.around( np.mean(y), decimals=decimals) ,np.around( np.std(x), decimals=decimals) ,np.around( np.std(y), decimals=decimals)

    covXY = np.around( covariance(x,y) , decimals=decimals )
    return meanx,meany,stdx,stdy,covXY

def plot_ms(x,y,xname='',yname='',unitsx='',unitsy='',addtitle='',zi=0,lenz=1,b=4.27,plotkk=False,doplot=True):

    if(plotkk):
        x2,x1 = np.max(x),np.min(x)
        y2,y1 = np.min(y),np.max(y)
        x_kk = np.linspace(x2,x1,100)
        a_kk = (y2-y1)/(x2-x1)
        y_kk = a_kk*x_kk + b
    
    meanx,meany,stdx,stdy,covXY = give_msc(x,y,decimals=2)

    #if (lenz==1): plt.suptitle(addtitle)
    if (doplot):
        plt.subplot(2,2*lenz,1+zi*2) #221
        plt.title(addtitle)
        plt.ylabel(yname+unitsy)
        plt.xlabel(xname+unitsx)
        if(plotkk): plt.plot(x_kk,y_kk,'k--')
        plt.scatter(x,y)
        
        plt.subplot(2,2*lenz,2+zi*2) #222
        plt.xlabel(yname+unitsy)
        plt.hist(y,color='b')
        plt.legend(loc=1, numpoints=1)
        
        plt.subplot(2,2*lenz,3+(zi+(lenz-1) )*2) #223
        plt.hist(x,color='g')
        plt.xlabel(xname+unitsx)
        plt.legend(loc=1, numpoints=1)

        plt.subplot(2,2*lenz,4 +(zi+(lenz-1) )*2) #224
        plt.plot(0,0,'b' ,label = yname+'='+str(meany)+'$\pm$'+str(stdy)+unitsy )
        plt.plot(0,0,'g' ,label = xname+'='+str(meanx)+'$\pm$'+str(stdx)+unitsx )
        plt.plot(0,0,'k' ,label = 'Cov['+xname+','+yname+'] = '+str(covXY) )
        plt.legend(loc='center', numpoints=1)
        plt.draw()

    return meanx,meany,stdx,stdy,covXY

def sigma_xy(x,y):
    ''' 
    returns sigma_xy , just covariance, no matrix 
    numpy.cov returns the whole covariance matrix
    numMath.sigma_xy(x,y) == np.cov(x,y)[0,1]
    '''
    meanx = np.mean(x)
    meany = np.mean(y)
    cov   = np.mean((x-meanx)*(y-meany))
    return cov

def corrc_xy(x,y):
    ''' 
    returns r_xy , just correlation coefficient, no matrix 
    numpy.corrcoeff returns the whole correlation coefficient matrix
    numMath.corrc_xy(x,y) == np.corrcoef(x,y)[0,1]
    '''
    num_sigma_xy = sigma_xy(x,y)
    num_corrc_xy = num_sigma_xy/(np.std(x)*np.std(y))
    return num_corrc_xy

def corrmat(covmat):
    dims = np.shape(covmat)
    cc = covmat*0.0

    for i in range(dims[0]):
        for j in range(dims[1]):
            cc[i][j] = covmat[i][j]/np.sqrt(covmat[i][i]*covmat[j][j])
    return cc

def stat2var(x,y):
    ''' return mx,sx,my,sy,cov(x,y),corr(x,y) '''
    return np.mean(x),np.std(x),np.mean(y),np.std(y),np.cov(x,y)[0,1],np.corrcoef(x,y)[0,1]

def covAverage(x,cov):
    """
    returns weight mean accoring to covariance matrix
    Gives the mean and std 
    accounting 
    for covariance matrix
    """
    Nx = len(x)
    W = np.ones(Nx)    
    invcov = linalg.inv(cov)
    var_x = 1./np.dot(W.T, np.dot(invcov,W))
    mean_x = var_x*np.dot(W.T, np.dot(invcov,x))
    std_x = np.sqrt(var_x)
    return(mean_x, std_x)

def plot3D_pierros(x,y,f_xy,zname='z=?',savename='plot3D_pierros.png',save=False):

    #X,Y = meshgrid(rS,rS)
    X,Y = x,y
    Z = f_xy
    print X , Y
    print Z
    print X.shape , Y.shape , Z.shape
    fig, ax = plt.subplots()
    plt.xlabel('$bias$',size=20)
    plt.ylabel('$\sigma_p\ [km\ s^{-1}]$',size=20)
    plt.suptitle(zname)
    #plt.zlabel('$Ratio(bias,\sigma_p) = R^{Distorted}_H(b,\sigma_p) / R_H$')
    #plt.yscale('log')
    #plt.xscale('log')
    #plt.ylim(1.99,2.5)
    #plt.xlim(55,67)
    #plt.plot(x,y)
    p = ax.pcolor(X, Y, Z, cmap=cm.jet, vmin=np.min(Z), vmax=np.max(Z) )#, label='$R^{(r)}_H/R^{(s)}_H$')
    #p = ax.pcolor(X, Y, Z, cmap=cm.jet, vmin=np.min(Z), vmax=np.max(Z),label='$\frac{R^{(r)}_H}{R^{(s)}_H}$')
    cb = fig.colorbar(p, ax=ax,label='$\mathcal{R}^{(RSD)}_H/\mathcal{R}^{(linear)}_H$ in $\%$')

    if save == True:
        print 'Saving ...'
        plt.savefig(savename+'.png',dpi=100)

def loadRand(fin,qsize=4,rbins=50,randsize=10):
    ''' 
    takes a string name of a file with data
    (col x rows) = (qsize x rbins) 
    returns an array(qsize x randsize) 
    with randomly sampling this file
    '''
    res = np.zeros([a.size,qsize])
    bsize = 0
    while bsize != randsize:
        a = np.random.random_integers(low=rbins, size=randsize)
        b = np.unique(a)
        bsize = b.size

    i=0
    with open(fin) as fd:
        for n, line in enumerate(fd):
            if (n in a):
                res[i] = np.fromstring(line.strip(),sep=' ')
                i = i + 1
    return res
    

def mySVD(matrix,doCheck=0,kk=False):
    '''
    Singular Value decomposition                                   
    method to compute the inverse
    of matrix        
    option: doCheck=0,1,2
    0: return only the inverted SVD matrix
    1: return previous plus composite matrices
    2: return previous plus checks
           0,1                         ,2
    USAGE: s,sU,sUt,sV,sVh,sSig,sinvSig,sCheck = numMath.mySVD(a,doCheck=2)
    '''
    ### Compute the SVD parts
    M,N       = matrix.shape
    U,s,Vh    = linalg.svd(matrix)
    Sig       = linalg.diagsvd(s,M,N)

    V = np.matrix(Vh).H
    Ut = np.matrix(U).T
    
    # invSig = linalg.inv( np.matrix(Sig) )
    invSig = np.matrix(Sig).I
    
    ### Correct for ill-ness
    w = np.where(Sig<=10**(-14))
    invSig[w]=0.0
    
    ### Compute the Inverse of the matrix 
    invSVD = V.dot(invSig).dot(Ut)  

    ### and check the matrix
    checkMatrix = U.dot(Sig.dot(Vh))
    checkProduct = invSVD.dot(matrix)

    #print ' invSVD = \n', invSVD
    #print ' sCheck = \n', checkMatrix

    #print ' invSVD.dot(matrix)     = \n  ', checkProduct

    if(doCheck==0):
        return(invSVD)
    elif(doCheck==1):
        return(invSVD,U,Ut,V,Vh,Sig,invSig)
    elif(doCheck==2):
        if(kk==True):  
            print 'V.dot(Vh)= \n',V.dot(Vh)
            print 'U.dot(Ut)= \n',U.dot(Ut)  
            print 'Sig.dot(invSig)= \n',Sig.dot(invSig)
        return(invSVD,U,Ut,V,Vh,Sig,invSig,checkMatrix)

    else:
        print'Read the description of numMath.mySVD'

################### Chi2 Robustness TEST for mocks ########################################### 
def Pierros_histogram(data_array,Normalization=True,numberBins=100):
    histo = np.histogram(data_array,bins=numberBins)
    x_center_hist = (histo[1][:-1]+histo[1][1:])/2
    y_hist = histo[0]
    yerror_hist = np.sqrt(y_hist)

    if Normalization==True:
        dx = histo[1][1] - histo[1][0]
        I_norm = np.sum(y_hist*dx)
        y_hist_normed = y_hist/I_norm
        error_hist_normed = yerror_hist/I_norm
        y_hist = y_hist_normed
        yerror_hist = error_hist_normed 
    return(x_center_hist,y_hist,yerror_hist)

def theoryChi2(x,pars):
    return( (1/(2**(pars[0]/2.)  * np.math.gamma(pars[0]/2.)) ) * x**(pars[0]/2. - 1) * np.exp( -x/2. ) )


def chi2_bias_test(chi2_mock,ndf=15,nbHistBins=4,method='minuit'):
    wok = np.where((chi2_mock<1600)&(chi2_mock>0))
    print np.max(chi2_mock)
    data = chi2_mock[wok]

    x_hist,y_hist,yerr_hist = Pierros_histogram(data,numberBins=nbHistBins) # 10

    guess = [np.float64(ndf)]
    res = fitting.dothefit(x_hist,y_hist,yerr_hist,guess,functname=theoryChi2,method=method)

    decimals = 2
    NDF_measured = np.around(res[1][0],decimals=decimals)
    dNDF_measured = np.around(res[2][0],decimals=decimals)
    chi2_measured = np.around(res[4],decimals=decimals)
    ndf_measured = np.around(res[5],decimals=decimals)

    # For plotting the theoretical chi2
    xmin,xmax = np.min(data),np.max(data) 
    xxx = np.linspace(xmin,xmax,1000)

    label_hist = "$ndf_{m} =$ "+str(NDF_measured)+"$\pm$"+str(dNDF_measured)+" at $\chi^2=$"+str(chi2_measured)+"/"+str(ndf_measured)
    plt.errorbar(x_hist,y_hist,yerr=yerr_hist,fmt='o',color='b',label=label_hist)
    plt.plot(xxx,theoryChi2(xxx,np.array([ndf])),'r-',label='ndf = '+str(ndf))
    plt.legend(loc=1)
################### END: Chi2 Robustness TEST for mocks ########################################### 

def pullCov(y,yth,cov):
    delta = y-yth
    eigenvals,rot = np.linalg.eigh(cov)
    newdelta = np.dot(rot.T,delta)
    pull = newdelta/np.sqrt( eigenvals )
    return pull

def pullDistr(x,x_err,x_mean=None,axis=None):
    """
    Tested on all zbins on Mocks North
    #1 pullN = ( x-np.average(x,weights=1./x_std**2) )/np.std(x, dtype=np.float64)
    #2 pullN = ( x-np.average(x,weights=1./x_std**2) )/x_std
    #3 pullN = ( x-np.mean(x) )/x_std
    1) -10e-13
    2) -10e-12
    3) -10e-13
    but the one ine use: -10e-14
    """
    if x_mean==None:
        if axis!=None:
            x_mean = np.mean(x,axis=axis)[:, None]
        else:
            x_mean = np.mean(x)
        pull = ( x-x_mean )/ ( x_err )
    else:
        pull = ( x-x_mean )/ ( x_err )
    return pull

def weight_mean(x_i,sig_i):

    weight_i = 1./sig_i**2.
    weight = sum(weight_i)
    weight_Norm_i = weight_i/weight

    mean_weight = sum( weight_i*x_i )

    return mean_weight   

def weight_std(x_i,sig_i):

    weight_i = 1./sig_i**2.
    weight = sum(weight_i)
    weight_Norm_i = weight_i/weight
    mean_weight = sum( weight_i*x_i )

    res = sum( weight_Norm_i*(x_i-mean_weight)**2. )
    
    return np.sqrt(res)

def NameZ2(minz,maxz,nbins):
    """
    returns 2 arrays which contain
    the edges of bins of redshift
    in double and string array 
    + 3 more arrays
    USE: zedge,znames,dz,zmid,nbins = numMath.NameZ2(minz=0.43,maxz=0.7,nbins=5)
    """
    zedge = np.linspace(minz,maxz,nbins+1)
    znames = np.array(np.zeros(nbins), dtype='|S20')
    
    for i in np.arange(nbins):
        znames[i]='z_'+str(zedge[i])+'_'+str(zedge[i+1])
    
    dz=np.zeros(nbins)+(zedge[1]-zedge[0])/2
    zmid=(zedge[np.arange(nbins)]+zedge[np.arange(nbins)+1])/2

    return(zedge,znames,dz,zmid,nbins)

def rebinMat(matrix):
    M,N = np.shape(matrix)
    return res

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def covsAvarage(x,cov):
    """
    Gives the mean and std 
    accounting 
    for covariance matrix
    """
    Nx = np.size(x)
    W = np.ones(Nx)    
    invcov = linalg.inv(cov)
    var_x = 1./np.dot(W.T, np.dot(invcov,W))
    mean_x = var_x*np.dot(W.T, np.dot(invcov,x))
    std_x = np.sqrt(var_x)
    return(mean_x, std_x)

def heaviside(x,x0=0,inverse=False):
    if inverse:
        delta = x0-x
    else:
        delta = x-x0
    res = 0.5*(np.sign(delta)+1)
    return res

def logBinning(r_max=1500.,nb_r=50,n_logint=10,a=1.):
    #a=1. #0.5                                                                                
    log_r_max=np.log10(r_max)
    powerL = np.zeros(nb_r)
    for i in xrange(nb_r):
            #radius[i] = 10**( ((i+a)-nb_r)/n_logint+log_r_max  )                             
            powerL[i] = ((i+a)-nb_r)/n_logint
            #radius[i]=(i+0.5)/(nb_r*(1./r_max)) # for regular binning                        
    radiusL=10**powerL * r_max
    return radiusL


def stat_realisations(datasim1,datasim2):
    """
    computes cross-covmat cross-corrmat
    for 2 variables 1 and 2
    coming from a simulation
    """
    dims=np.shape(datasim1)
    nsim=dims[1]
    nbins=dims[0]
    meansim1=np.zeros(nbins)
    sigsim1=np.zeros(nbins)
    meansim2=np.zeros(nbins)
    sigsim2=np.zeros(nbins)
    for i in np.arange(nbins):
        meansim1[i]=np.mean(datasim1[i,:])
        sigsim1[i]=np.std(datasim1[i,:])
    for i in np.arange(nbins):
        meansim2[i]=np.mean(datasim2[i,:])
        sigsim2[i]=np.std(datasim2[i,:])

    print '   stat:do covmat'
    covmat=np.zeros((nbins,nbins))
    for i in np.arange(nbins):
        for j in np.arange(nbins):
            covmat[i,j]=np.mean((datasim1[i,:]-meansim1[i])*(datasim2[j,:]-meansim2[j]))

    print '   stat:do cormat'
    cormat=np.zeros((nbins,nbins))
    for i in np.arange(nbins):
        for j in np.arange(nbins):
            cormat[i,j]=covmat[i,j]/np.sqrt(covmat[i,i]*covmat[j,j])

    return(meansim1,sigsim1,meansim2,sigsim2,covmat,cormat)

def plotCorrMat(x,y,datasim1,datasim2,savename='corrplot',save=False):
    print '  do stat'
    meansim1,sigsim1,meansim2,sigsim2,covmat,cormat = stat_realisations(datasim1,datasim2)
    
    #X,Y = meshgrid(rS,rS)
    X,Y = np.sort(x),np.sort(y)
    Z = cormat
    print X , Y
    print Z
    print X.shape , Y.shape , Z.shape
    fig, ax = plt.subplots()
    plt.ylabel('$bias$')
    plt.xlabel('$\mathcal{R}_H$ ($h^{-1}\ Mpc$)')
    ## plt.yscale('log')
    ## plt.xscale('log')
    #plt.ylim(1.99,2.5)
    #plt.xlim(55,67)
    plt.plot(x,y)
    p = ax.pcolor(X, Y, Z, cmap=cm.RdBu, vmin=Z.min(), vmax=Z.max())
    cb = fig.colorbar(p, ax=ax)
    if save == True:
        print 'Saving ...'
        plt.savefig(savename+'.png',dpi=100)

# rounding indices inside a dictionary
class LessPrecise(float):
    def __repr__(self):
        return str(self)

def roundDict(d,num=4):
    for k, v in d.items():
        if isinstance( v, np.str ): pass
        else:
            v = LessPrecise(np.round(v, num))
            d[k] = v
    return d 
# rounding indices inside a dictionary
