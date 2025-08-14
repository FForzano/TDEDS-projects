import numpy as np
import tikzplotlib as tpl
import random, math
from matplotlib import pyplot as plt

def is_Cth(b,threeshold):
    for bin in b:
        if bin > threeshold:
            return True
    return False

def TCS_decisor(b,threeshold):
    for i in range(len(b)):
        if b[i] > threeshold:
            return i
    return -1

def MBS_decisor(b):
    return np.argmax(b)

def success_probability_sim(Nq = 100, Nsb = 5, mu = 0, sigma2 = 1, P = 100, SNR = 3, threeshold = 4, criterium = 'both'):
    A = math.sqrt(SNR*sigma2)

    # Nb = Nq/Nsb
    Nb = int(Nq/Nsb) # Number of bins

    # TNR = threeshold**2/sigma2
    # threeshold = math.sqrt(TNR*sigma2)

    success_TCS = 0
    success_MBS = 0

    for current_P in range(P):
        n = np.zeros(Nq)
        for i in range(Nq):
            n[i] = random.gauss(mu, math.sqrt(sigma2))
        
        x = np.copy(n)
        q_star = int(random.random()*Nq)
        x[q_star] += A
        correct_bin = int(q_star/Nsb) # for success check

        b = np.zeros(Nb) # bin vector
        for i in range(Nb):
            b[i] = np.sum(x[i*Nsb:(i+1)*Nsb]**2)
        
        '''
        plt.hist(b)
        plt.plot(np.full(Nb,threeshold))
        plt.show()
        '''

        if is_Cth(b,threeshold):

            if criterium == 'both':
                i_hat_TCS = TCS_decisor(b,threeshold)
                i_hat_MBS = MBS_decisor(b)

                # Valuation TCS
                if i_hat_TCS == correct_bin:
                    success_TCS += 1

                # Valuation MBS
                if i_hat_MBS == correct_bin:
                    success_MBS += 1
            
            elif criterium == 'TCS':
                i_hat_TCS = TCS_decisor(b,threeshold)

                # Valuation TCS
                if i_hat_TCS == correct_bin:
                    success_TCS += 1

            elif criterium == 'MBS':
                i_hat_TCS = MBS_decisor(b)

                # Valuation MBS
                if i_hat_MBS == correct_bin:
                    success_MBS += 1

    if criterium == 'both':
        return success_TCS/P, success_MBS/P
    elif criterium == 'TCS':
        return success_TCS/P
    elif criterium == 'MBS':
        return success_MBS/P


if __name__ == "__main__":
    #Nq = 100 # number of samples for the all signal
    #Nsb = 5 # Numbers of samples for bin
    #mu = 0
    sigma2 = 1
    P = 10000

    ### SIMULATION 2: varing SNR and threeshold, evaluate the probability of success ###

    SNR = np.linspace(0,25,50) # dB 
    SNR = 10**(SNR/10) # linear value

    threeshold = np.array([5,10,15,20]) # dB
    threeshold = 10**(threeshold/10) # linear values

    # Può darsi sia così perchè non è potenza ??? (Ma in realtà sì...)
    # threeshold = np.array([10,20,30,40]) # dB
    # threeshold = 10**(threeshold/20) # linear values

    # Observation: i use the threshold directly because the noise has unitary power, so rho = threshold**2

    for curr_th in threeshold:
        pdf_rho_TCS = []
        pdf_rho_MBS = []

        for curr_SNR in SNR:
            P_TCS, P_MBS = success_probability_sim(SNR=curr_SNR, threeshold=curr_th, criterium='both', P=P)
            pdf_rho_TCS.append(P_TCS)
            pdf_rho_MBS.append(P_MBS)

        pdf_rho_TCS = np.array(pdf_rho_TCS)
        pdf_rho_MBS = np.array(pdf_rho_MBS)

        
        # plt.title("Threshold crossing search")
        # plt.plot(10*np.log10(SNR),pdf_rho_TCS, label='TNR = '+str(2*curr_th)+' dB')
        # plt.xlabel("SNR [db]")
        # plt.ylabel("Success probability")
        
        
        plt.title("Maximum bin search")
        plt.plot(10*np.log10(SNR),pdf_rho_MBS, label='TNR = '+str(2*curr_th)+' dB') # th^2 --> 2*th[dB] (sigma2 = 1 = 0dB)
        plt.xlabel("SNR [dB]")
        plt.ylabel("Success probability") 


    plt.legend()
    tpl.save("maximum_bin_search.tex")
    plt.show()

    
