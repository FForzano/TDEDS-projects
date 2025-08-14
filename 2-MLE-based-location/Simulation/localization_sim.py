from cProfile import label
import numpy as np
import random, math
from cartesianspace import cartesianspace
from matplotlib import pyplot as plt
import tikzplotlib as tpl
from scipy.linalg import norm

from MLestimator import ML_estimation, ML_estimation_unbias

'''
Avviare per la simulazione vera con delta=0.2 e P>=5000 
'''

if __name__ == "__main__":
    P = 10000 # P is the number of agent position considered
    L = 10 # L = 50m --> length of the side of the square
    delta =  L/200 # space discetization parameter

    p_LOS = 0.7
    # p_NLOS = 1 - p_LOS

    sigma2 = np.linspace((L/100)**2,((L/100*10)**2),10)
    b_par = L/100
    b = [b_par*0, b_par*2, b_par*4, b_par*8]

    # The space grid is [-L/2,L/2]x[-L/2,L/2] with discretization parameter delta.
    # The four anchors are in the four corners.

    grid = cartesianspace(L, delta)
    axis_npoint = grid.get_npoint()

    # anchor1:              anchor2:            anchor3             anchor4:
    #     . . . . . .           * . . . . .         . . . . . *         . . . . . .
    #     . . . . . .           . . . . . .         . . . . . .         . . . . . .
    #     . . . . . .           . . . . . .         . . . . . .         . . . . . .
    #     . . . . . .           . . . . . .         . . . . . .         . . . . . .
    #     . . . . . .           . . . . . .         . . . . . .         . . . . . .
    #     * . . . . .           . . . . . .         . . . . . .         . . . . . *

    anchors = []
    anchors.append( (grid.axis()[0], grid.axis()[0]) )
    anchors.append( (grid.axis()[0], grid.axis()[-1]) )
    anchors.append( (grid.axis()[-1], grid.axis()[-1]) )
    anchors.append( (grid.axis()[-1], grid.axis()[0]) )
    anchors = np.array(anchors)

    # all possible distances from the anchors:
    positions_grid = grid.get_grid()
    possibles_d = []
    for anchor in anchors:

        # all possible distances --> ML estimator have to choise the most likelihood
        value = np.empty((), dtype=object)
        value[()] = anchor
        possibles_d.append(positions_grid - np.full(np.shape(positions_grid), value, dtype=object))

        i, j = 0, 0
        for i in range(grid.get_npoint()):
            for j in range(grid.get_npoint()):
                possibles_d[-1][i,j] = norm(possibles_d[-1][i,j])
                j += 1
            i += 1

    # print("Simulate the biased algorithm")
    # # BIASED SIMULATION
    # for current_b in b:

    #     MSE = []

    #     for current_sigma2 in sigma2:

    #         square_errors = []

    #         for i in range(P):

    #             x = random.choice( grid.axis() )
    #             y = random.choice( grid.axis() )

    #             agent_position = (x,y)

    #             real_d = []
    #             estimated_d = []
    #             bias = []
    #             for anchor in anchors:
    #                 # real distances
    #                 real_d.append( np.full(np.shape(positions_grid), norm(agent_position - anchor)) )
    #                 bias_value = np.random.binomial(1, 1-p_LOS)*current_b
    #                 bias.append( np.full(shape=np.shape(possibles_d[-1]), fill_value=bias_value ) )
    #                 estimated_d.append( real_d[-1] + bias[-1] + np.random.normal( 0, math.sqrt(current_sigma2), size=(np.shape(possibles_d[-1])) ) )



    #             # estimated_position = ML_estimation(
    #             #     real_position=agent_position,
    #             #     p_LOS=p_LOS,
    #             #     sigma2=current_sigma2,
    #             #     anchors = anchors,
    #             #     space=grid,
    #             #     b=b )
    #             estimated_position = ML_estimation(
    #                 estimated_d = estimated_d,
    #                 possibles_d = possibles_d,
    #                 position_grid = positions_grid,
    #                 space=grid )
                
    #             square_errors.append( sum((np.array(agent_position) - np.array(estimated_position))**2) )
    
    #         MSE.append( np.sum(square_errors)/P )
    #         print("#")

    #     plt.plot(sigma2,MSE, label='b = ' + str(current_b))
    #     print('\n')

                

    # plt.xlabel('sigma2')
    # plt.ylabel('MSE')
    # plt.legend()
    # tpl.save("MSE_biased_def.tex")
    # plt.show()
    # plt.close()
    # print("Biased plot done.")

    print("Simulate the unbiased algorithm")
    # UNBIASED SIMULATION
    for current_b in b:

        MSE = []

        for current_sigma2 in sigma2:

            square_errors = []

            for i in range(P):

                x = random.choice( grid.axis() )
                y = random.choice( grid.axis() )

                agent_position = (x,y)

                real_d = []
                estimated_d = []
                bias = []
                for anchor in anchors:
                    # real distances
                    real_d.append( np.full(np.shape(positions_grid), norm(agent_position - anchor)) )
                    bias_value = np.random.binomial(1, 1-p_LOS)*current_b
                    bias.append( np.full(shape=np.shape(possibles_d[-1]), fill_value=bias_value ) )
                    estimated_d.append( real_d[-1] + bias[-1] + np.random.normal( 0, math.sqrt(current_sigma2), size=(np.shape(possibles_d[-1])) ) )



                # estimated_position = ML_estimation(
                #     real_position=agent_position,
                #     p_LOS=p_LOS,
                #     sigma2=current_sigma2,
                #     anchors = anchors,
                #     space=grid,
                #     b=b )
                estimated_position = ML_estimation_unbias(
                    estimated_d = estimated_d,
                    possibles_d = possibles_d,
                    position_grid = positions_grid,
                    space=grid,
                    bias=bias,
                    sigma2 = current_sigma2,
                    p_LOS = p_LOS )
                
                square_errors.append( sum((np.array(agent_position) - np.array(estimated_position))**2) )
    
            MSE.append( np.sum(square_errors)/P )
            print("#", end=' ')

        plt.plot(sigma2,MSE, label='b = ' + str(current_b))
        print('\n')

                

    plt.xlabel('sigma2')
    plt.ylabel('MSE')
    plt.legend()
    tpl.save("MSE_unbiased_def.tex")
    plt.show()