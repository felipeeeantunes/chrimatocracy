import numpy as np

def Fun(X, gamma, eta0):
    delta  = np.log(0.01)
    etaMax = np.log(max(X.values))
    
    A= 1+(eta0/(etaMax-delta))**gamma
    B= 1+(eta0/(np.log(X.values)-delta))**gamma
    
    f = A/B
    return f
# This generates a random number with distribution according to eq 6 in the manuscript


def drand_distr(gamma, csi0, csim=100., delt=0.):
    F = np.random.uniform(0,1)
    bla = csi0/(csim-delt)
    blabla = F/(bla**gamma+1-F)
    
    return np.exp(csi0*blabla**(1./gamma)+delt)

# this is F(x) in eq. 5
def func(x, gamma, csi0, csim=100., delt=0.):
    # cumulative distribution
    num1 = np.log(x)-delt
    num2 = (csi0/num1)**gamma
    num3 = (csi0/(csim-delt))**gamma
    term2 = 1.+num3
    term3 = 1./(num2+1.)
    return term2*term3

# this is f(x) in eq. 6
def func2(x, gamma, csi0, csim=100., delt=0.):
    # distribution
    num1 = np.log(x)-delt
    num2 = (csi0/num1)**gamma
    num3 = (csi0/(csim-delt))**gamma
    term1 = gamma/x
    term2 = 1.+num3
    term3 = num2/( (num2+1.)**2 )
    term3 /= num1
    return term1*term2*term3

    

# This evaluates lnL and its gradient
def lkhd_distr(nums, gamma, csi0, csim, delt):
    sum1 = 0.
    sum2 = 0.
    sum3 = 0.
    sum4 = 0.
    sum5 = 0.
    sum6 = 0.
    N = len(nums)
    num1 = csi0/(csim-delt)
    num1g = num1**gamma
    termb = num1g/(1.+num1g)
    for ele in nums:
        num2 = np.log(ele)
        num3 = num2-delt
        num4 = csi0/(num3)
        num4g = num4**gamma
        term1 = num4g/(num4g+1.)
        # for lkhd
        sum1 += num2
        sum2 += np.log(num3)
        sum3 += np.log(1.+num4g)
        # for dlnldcsi0
        sum4 += term1
        # for dlnldgamma
        sum5 += np.log(num4)*term1
    lkhd = N*(np.log(gamma)+np.log(1.+num1g)+gamma*np.log(csi0))-sum1-(gamma+1.)*sum2-2.*sum3
    g1 = N*(gamma*termb/csi0 + gamma/csi0)-2.*gamma*sum4/csi0 #csi0
    g2 = -N*gamma*termb/(csim-delt) #csim
    g3 = N*(1./gamma + termb*np.log(num1)+np.log(csi0)) - sum2-2.*sum5 #gamma
    return lkhd, (g1, g2, g3)


### This algorithm may be unstable (never finishes) for some small sets of numbers For the ones in the manuscript it is working fine
# This evaluates parameters for obtaining maximum value of lnL
def maxLKHD_distr(nums, gamma=3., csi0=3., csim=100., delt=0.01, lamb=1., eps=1.e-4):
    lkhd, grad = lkhd_distr(nums, gamma, csi0, csim, delt)
    norm = (grad[0]**2+grad[1]**2+grad[2]**2)**.5
    ncsi0 = csi0+lamb*grad[0]/norm
    ncsim = csim#+lamb*grad[1]/norm
    ngamma = gamma+lamb*grad[2]/norm
    nlkhd, ngrad = lkhd_distr(nums, ngamma, ncsi0, ncsim, delt)
    while lamb*norm > eps:
        #print lamb, norm
        if nlkhd>lkhd: # accepted
            grad = ngrad
            csi0 = ncsi0
            csim = ncsim
            gamma = ngamma
            lkhd = nlkhd
            lamb *= 1.2
        else:
            lamb *= .01
        norm = (grad[0]**2+grad[1]**2+grad[2]**2)**.5
        ncsi0 = csi0+lamb*grad[0]/norm
        ncsim = csim#+lamb*grad[1]/norm
        ngamma = gamma+lamb*grad[2]/norm
        nlkhd, ngrad = lkhd_distr(nums, ngamma, ncsi0, ncsim, delt)
    return gamma, csi0, csim