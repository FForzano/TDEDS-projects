import numpy as np
from scipy.stats import ncx2
import random, math
from matplotlib import pyplot as plt
import tikzplotlib as tpl


def bin_simulation(Nq = 100, Nsb = 5, mu = 0, sigma2 = 1, P = 100, SNR = 3):
    A = math.sqrt(SNR*sigma2)

    # Nb = Nq/Nsb
    Nb = int(Nq/Nsb) # Number of bins

    # TNR = threeshold**2/sigma2
    # threeshold = math.sqrt(TNR*sigma2)

    signal_bin_occurences = []
    onlynoise_bin_occurences = []

    for current_P in range(P):
        n = np.zeros(Nq)
        for i in range(Nq):
            n[i] = random.gauss(mu, math.sqrt(sigma2))
        
        x = np.copy(n)
        q_star = 0
        x[q_star] += A
        correct_bin = int(q_star/Nsb) # for success check

        b = np.zeros(Nb) # bin vector
        # Il bin 0 contiene il segnale, i bin successivi no
        for i in range(Nb):
            b[i] = np.sum(x[i*Nsb:(i+1)*Nsb]**2)
        
        signal_bin_occurences.append(b[0])
        onlynoise_bin_occurences.append(b[1])
    
    return onlynoise_bin_occurences, signal_bin_occurences
    
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
        return A**2/sigma2 *Np

if __name__ == "__main__":

    Nsb = 5 # Numbers of samples for bin
    Nq = Nsb*2 # number of samples for the all signal --> we need only 2 bin for this simulation
    Np = 1 # Number of signal s(t) sent
    P=100000 # Number of realizations for the simulation
    mu = 0
    sigma2 = 1

    ### SIMULATION 1: fix SNR and plot the differences between the theorical PDF of a bin and the simulative one ###

    SNR = 10 # dB 
    SNR = 10**(SNR/10) # linear value

    noise_bin_realizations, signal_bin_realizations = bin_simulation(SNR=SNR, P=P, sigma2=sigma2, mu=mu)

    A = math.sqrt(SNR*sigma2)
    theorical_noise_bin = sigma2/Np * ncx2.pdf(np.linspace(0,50,1000),Np*Nsb,lambda_parameter(0,sigma2,Np,Nsb))
    theorical_signal_bin = sigma2/Np * ncx2.pdf(np.linspace(0,50,1000),Np*Nsb,lambda_parameter(A,sigma2,Np,Nsb))

    plt.title("Only noise bin PDF")
    plt.xlabel("b")
    plt.ylabel("f_B(b)")
    plt.plot(np.linspace(0,50,1000), theorical_noise_bin)
    plt.hist(noise_bin_realizations, bins=50, density=True, edgecolor = 'black')
    tpl.save("noise_bin1.tex")
    plt.show()

    plt.title("Signal bin PDF")
    plt.xlabel("b")
    plt.ylabel("f_B(b)")
    plt.plot(np.linspace(0,50,1000) ,theorical_signal_bin)
    plt.hist(signal_bin_realizations, bins=50, density=True, edgecolor = 'black')
    tpl.save("signal_bin1.tex")
    plt.show()

    
