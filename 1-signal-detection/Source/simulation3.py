import numpy as np
from scipy.stats import ncx2
import math
from matplotlib import pyplot as plt
import tikzplotlib as tpl

    
def lambda_parameter(A,sigma2,Np,Nsb):
    '''
    This function return the lambda parameter. 
    The function work only for signal bin with A amplitude of signal and sigma2 power of noise
    (for only noise bin A=0) with only one sample which contains the signal (only one x[i] in 
    the Nsb samples for bin).
    '''

    if A == 0:
        return 0
    else:
        return ((A**2)/sigma2 )*Np

if __name__ == "__main__":

    Nsb = 5 # Numbers of samples for bin
    Nq = Nsb*2 # number of samples for the all signal --> we need only 2 bin for this simulation
    Np = 1 # Number of signal s(t) sent
    P=100000 # Number of realizations for the simulation
    mu = 0
    sigma2 = 1

    SNR = 20 # dB 
    SNR = 10**(SNR/10) # linear value


    bin_values = 10**(np.linspace(-10,20,1000)/10)

    A = math.sqrt(SNR*sigma2)
    theorical_noise_bin = sigma2/Np * ncx2.pdf(bin_values,Np*Nsb,lambda_parameter(0,sigma2,Np,Nsb))
    theorical_signal_bin = sigma2/Np * ncx2.pdf(bin_values,Np*Nsb,lambda_parameter(A,sigma2,Np,Nsb))

    plt.title("Threshold plot")
    plt.xlabel("b")
    plt.ylabel("f_B(b)")
    plt.plot(bin_values, theorical_noise_bin, label='noise bin')
    # plt.fill_between(bin_values, theorical_noise_bin, alpha=0.5, linewidth=1)
    plt.plot(bin_values, theorical_signal_bin, label='signal bin')
    # plt.fill_between(bin_values ,theorical_signal_bin, alpha=0.5)

    i = 0
    TNR_dB = [5,10,15,20]
    for threshold in 10**(np.array([5,10,15,20])/10):
        plt.plot(np.array([threshold,threshold]), [0,0.155], label='TNR = ' + str(TNR_dB[i]) + 'dB')
        i += 1

    plt.legend()
    tpl.save("threshold_plot.tex")
    plt.show()


