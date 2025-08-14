import numpy as np
from numpy import diff
from scipy.linalg import sqrtm

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def QMSE(rho, sigma):
    differences = rho - sigma
    trace_distances = []
    for i in range(np.shape(differences)[2]):
        trace_distances.append(0.5*np.trace(sqrtm(differences[:,:,i].conj().T @ differences[:,:,i])) )

    trace_distances = np.array(trace_distances)
    trace_distances = np.real(trace_distances)
    return np.sum(trace_distances**2)/(len(trace_distances)-1)

def CMSE(data_mat1, data_mat2):
    differences = data_mat1 - data_mat2
    square_diff = np.real(np.sum(differences * differences.conj(), axis=0))
    return np.sum(square_diff)/(len(square_diff)-1)
