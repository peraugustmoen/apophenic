### METHODS FOR LD50

# Import
import numpy as np
import math
import numba as nb


# Methods
@nb.njit(nb.float64(nb.float64))
def get_phi(x):
    return math.exp(-x**2 / 2) / math.sqrt(2*np.pi)


@nb.njit(nb.float64(nb.float64))
def get_Phi(x):
    return .5*math.erfc(-x/np.sqrt(2))


@nb.njit(nb.float64(nb.float64))
def get_one_minus_Phi(x):
    return .5*math.erfc(x/np.sqrt(2))


@nb.njit(nb.float64[:](nb.float64[:]))
def get_phi_array(eta):
    return np.exp(-eta**2 / 2) / np.sqrt(2*np.pi)


@nb.njit(nb.float64[:](nb.float64[:]))
def get_Phi_array(x):
    n = len(x)
    cdfs = np.zeros(n)
    for i in range(n):
        cdfs[i] = get_Phi(x[i])
    return cdfs


@nb.njit(nb.float64[:](nb.float64[:]))
def get_one_minus_Phi_array(x):
    n = len(x)
    one_min_cdfs = np.zeros(n)
    for i in range(n):
        one_min_cdfs[i] = get_one_minus_Phi(x[i])
    return one_min_cdfs


@nb.njit(nb.float64[:](nb.float64[:], nb.int64, nb.float64[:]))
def rnd1(x, decimals, out):
    return np.round_(x, decimals, out)


@nb.njit(nb.types.Tuple((nb.float64[:], nb.float64[:]))(nb.int64, nb.float64, nb.float64, nb.float64, nb.float64))
def bruceton(n, x0, d, mu0, sigma0):
    x = np.zeros(n)
    y = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            x[i] = x0
        else:
            x[i] = x[i-1] + (1 - 2*y[i-1])*d
        y[i] = (sigma0*np.random.randn() + mu0) <= x[i]
    return rnd1(x, 2, x), y


@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:], nb.int64, nb.float64, nb.float64))
def fit_probit(x, y, max_iter=50, threshold=1e-10, step_size=.1):
    n = len(x)
    params = np.zeros(2)
    for k in range(max_iter):
        prev_params = params.copy()
        
        Fisher = np.zeros((2, 2))
        gradient = np.zeros(2)
        
        for i in range(n):
            eta = params[0] + params[1]*x[i]
            phi = get_phi(eta)
            Phi = get_Phi(eta)
            one_minus_Phi = get_one_minus_Phi(eta)
            
            if Phi*one_minus_Phi == 0:
                return np.inf*np.ones(2)
            
            gradient += phi*(y[i] - Phi)/(Phi*one_minus_Phi) * np.array([1, x[i]])
            Fisher += phi**2/(Phi*one_minus_Phi) * np.array([[1, x[i]], [x[i], x[i]**2]])
        try:
            params += step_size * np.linalg.inv(Fisher) @ gradient
        except:
            return np.inf*np.ones(2)
        diff = (params[0] - prev_params[0])**2 + (params[1] - prev_params[1])**2
        if diff < threshold:
            return params
    return np.inf*np.ones(2)


@nb.njit(nb.float64[:,:](nb.float64, nb.float64, nb.int64, nb.float64, nb.float64, nb.int64))
def estimate_MC_fisher(alpha, beta, n, x0, d, T):
    result = np.zeros((2, 2))
    mu0 = -alpha/beta
    sigma0 = 1/beta
    for t in range(T):
        xs, ys = bruceton(n, x0, d, mu0, sigma0)
        
        l_aa = 0
        l_ab = 0
        l_bb = 0
        
        for i in range(n):
            x = xs[i]
            y = ys[i]
            eta = alpha + beta*x
            phi = get_phi(eta)
            Phi = get_Phi(eta)
            one_minus_Phi = get_one_minus_Phi(eta)
            
            if y == 1:
                term = ((-eta)*phi*Phi - phi**2)/Phi**2
            if y == 0:
                term = -((-eta)*phi*one_minus_Phi + phi**2)/one_minus_Phi**2
            
            l_aa += term
            l_ab += term * x
            l_bb += term * x**2
            
        result[0][0] -= l_aa
        result[0][1] -= l_ab
        result[1][0] -= l_ab
        result[1][1] -= l_bb
    
    return result/T
