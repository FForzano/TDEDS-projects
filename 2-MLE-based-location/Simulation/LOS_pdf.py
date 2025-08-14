import numpy as np
import random
from cartesianspace import cartesianspace
from matplotlib import pyplot as plt
import tikzplotlib as tpl

if __name__ == "__main__":
    P = 10000 # P is the number of agent position considered
    L = 10
    b = L/100

    p_LOS = 0.7


    bias = []
    for i in range(P):
        bias_value = np.random.binomial(1, 1-p_LOS)*b
        bias.append( bias_value )

    plt.hist(bias, bins=5, density=False)
    locs, _ = plt.yticks() 
    # print(locs)
    plt.yticks(locs,np.round(locs/len(bias),3))
    # count, data = np.histogram(bias)
    # count_norm = count / P
    # plt.hist(count_norm, np.array(list(data)+[1.1]))
    plt.xlabel("b")
    plt.ylabel("PMF")
    tpl.save("bias_pmf.tex")
    plt.show()