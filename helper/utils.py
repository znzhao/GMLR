import os
import time
import numpy as np
import numba as nb
import datetime
from numba import njit

colormap = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:blue', 
            5:'tab:purple', 6:'tab:brown', 7:'tab:pink', 8:'tab:gray', 9:'tab:olive', 10:'tab:cyan'}

class Timer(object):
    def __init__(self, name=None, display = True):
        self.name = name
        self.display = display

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name and self.display:
            print('[{}] '.format(self.name), end='\t')
        if self.display: print('Elapsed in {:.4f} secs.'.format(time.time() - self.tstart))


def gradient(x:np.array, func):
    '''
    Compute the gradient of a given function with respect to its variables.

    Args:
        x (np.array): Input vector.
        func (function): The function for which the gradient needs to be computed.

    Returns:
        np.array: Gradient of the function with respect to each variable in x.
    '''
    x = x.astype(float)
    N = x.shape[0]
    gradient = []
    for i in range(N):
        eps = abs(x[i]) * np.finfo(np.float32).eps
        xx0 = 1. * x[i]
        f0 = func(x)
        x[i] = x[i] + eps
        f1 = func(x)
        gradient.append(np.array([f1 - f0])/eps)
        x[i] = xx0
    return np.array(gradient).reshape(tuple(list(x.shape) + list(np.array(func(x)).shape)))

def hessian(x:np.array, func):
    '''
    Compute the Hessian matrix of a given function.

    Args:
        x (np.array): Input vector.
        func (function): The function for which the Hessian matrix needs to be computed.

    Returns:
        np.array: Hessian matrix of the function.
    '''
    x = x.astype(float)
    N = x.shape[0]
    hessian = np.zeros(tuple([N, N]+ list(func(x).shape))) 
    gd_0 = gradient( x, func)
    eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps
    for i in range(N):
        xx0 = x[i]
        x[i] = xx0 + eps
        gd_1 =  gradient(x, func)
        hessian[:,i] = ((gd_1 - gd_0)/eps).reshape(tuple(list(x.shape) + list(func(x).shape)))
        x[i] = xx0
    return hessian


def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (red, green, blue)

def interpolateColor(color1, color2, alpha):
    color1 = hex_to_rgb(color1)
    color2 = hex_to_rgb(color2)
    newcolor = tuple(np.array(color1)*(1-alpha) + np.array(color2)*alpha)
    return rgb_to_hex(int(newcolor[0]), int(newcolor[1]), int(newcolor[2]))

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            fullname = fullname.replace("\\","/")
            yield fullname

def nextBusinessDay(date):
    if date.weekday() == 4:
        date = date + datetime.timedelta(days=3)
    else:
        date = date + datetime.timedelta(days=1)
    return date

def nextKBusinessDay(date, k):
    if k == 1:
        return nextBusinessDay(date)
    else:
        return nextKBusinessDay(nextBusinessDay(date), k-1)

def maxDrawDown(data):
    data = np.cumsum(data)
    max_so_far = data[0]
    drawdowns = []
    for x in data:
        max_so_far = max(max_so_far, x)
        drawdowns.append(max_so_far - x) 
    return max(drawdowns)

def maxDrawUp(data):
    data = np.cumsum(data)
    min_so_far = data[0]
    drawdowns = []
    for x in data:
        min_so_far = min(min_so_far, x)
        drawdowns.append(x - min_so_far) 
    return max(drawdowns)

def sortinoErr(data):
    err = [min(x, 0) for x in data]
    return np.std(err)

def calPnLs(data):
    return np.sum(data)

def calSortino(data):
    PnLs = calPnLs(data)
    err = sortinoErr(data)
    return PnLs / err

@nb.extending.overload(np.array)
def np_array_ol(x):
    """Numba JIT-compiled function for deep copying arrays."""
    if isinstance(x, nb.types.Array):
        def impl(x):
            return np.copy(x)
        return impl

def pyObjToNumbaObj(pyObj):
    if type(pyObj)==dict:
        pyDict=pyObj
        keys=list(pyDict.keys())
        values=list(pyDict.values())
        if type(keys[0]) == str:nbhKeytype = nb.types.string
        elif type(keys[0]) == int:nbhKeytype = nb.types.int32
        elif type(keys[0]) == float: nbhKeytype = nb.types.float32
        if type(values[0])==int:
            nbh=nb.typed.Dict.empty(nbhKeytype,nb.types.int32)
            for i,key in enumerate(keys):nbh[key]=values[i]
            return(nbh)
        elif type(values[0])==str:
            nbh=nb.typed.Dict.empty(nbhKeytype,nb.types.string)
            for i,key in enumerate(keys):nbh[key]=values[i]
            return(nbh)
        elif type(values[0]) == float:
            nbh=nb.typed.Dict.empty(nbhKeytype,nb.types.float32)
            for i,key in enumerate(keys):nbh[key]=values[i]
            return(nbh)
        elif type(values[0])==dict:
            for i,subDict in enumerate(values):
                subDict=pyObjToNumbaObj(subDict)
                if i==0:nbh=nb.typed.Dict.empty(nbhKeytype,nb.typeof(subDict))
                nbh[keys[i]]=subDict
            return(nbh)
        elif type(values[0])==list:
            for i, subList in enumerate(values):
                subList=pyObjToNumbaObj(subList)
                if i==0:nbh=nb.typed.Dict.empty(nbhKeytype,nb.typeof(subList))
                nbh[keys[i]] = subList
            return(nbh)
    elif type(pyObj)==list:
        pyList=pyObj
        data = pyList[0]
        if type(data) == int:
            nbs = nb.typed.List.empty_list(nb.types.int32)
            for data_ in pyList: nbs.append(data_)
            return (nbs)
        elif type(data) == str:
            nbs = nb.typed.List.empty_list(nb.types.string)
            for data_ in pyList: nbs.append(data_)
            return (nbs)
        elif type(data) == float:
            nbs = nb.typed.List.empty_list(nb.types.float32)
            for data_ in pyList: nbs.append(data_)
            return (nbs)
        elif type(data) == dict:
            for i,subDict in enumerate(pyList):
                subDict = pyObjToNumbaObj(subDict)
                if i == 0: nbs = nb.typed.List.empty_list(nb.typeof(subDict))  
                nbs.append(subDict)
            return(nbs)
        elif type(data) == list:
            for i,subList in enumerate(pyList):
                subList=pyObjToNumbaObj(subList)
                if i==0: nbs = nb.typed.List.empty_list(nb.typeof(subList))  
                nbs.append(subList)
            return(nbs)

@njit
def dot3d(X: np.array, betas: np.array):
    res = np.zeros((X.shape[0], betas.shape[2], betas.shape[0]))
    for g in range(betas.shape[0]):
        res[:,:,g] = X.dot(betas[g])
    return res

if __name__ == "__main__":
    print('Data Tools loaded.')