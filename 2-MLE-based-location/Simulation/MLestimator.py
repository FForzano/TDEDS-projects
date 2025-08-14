import numpy as np
from scipy.linalg import norm
from cartesianspace import cartesianspace
import math


# def ML_estimation(real_position, anchors, space, p_LOS=0.5, sigma2=1, b=1):
def ML_estimation(estimated_d, possibles_d, position_grid, space):
    # positions_grid = space.get_grid()

    # real_d = []

    # possibles_d = []
    # estimated_d = []
    # bias = []
    # for anchor in anchors:
    #     # real distances
    #     real_d.append( np.full(np.shape(positions_grid), norm(real_position - anchor)) )

    #     # all possible distances --> ML estimator have to choise the most likelihood
    #     value = np.empty((), dtype=object)
    #     value[()] = anchor
    #     possibles_d.append(positions_grid - np.full(np.shape(positions_grid), value, dtype=object))

    #     i, j = 0, 0
    #     for i in range(space.get_npoint()):
    #         for j in range(space.get_npoint()):
    #             possibles_d[-1][i,j] = norm(possibles_d[-1][i,j])
    #             j += 1
    #         i += 1

    #     bias_value = np.random.binomial(1, 1-p_LOS)*b
    #     bias.append( np.full(shape=np.shape(possibles_d[-1]), fill_value=bias_value ) )
    #     estimated_d.append( real_d[-1] + bias[-1] + np.random.normal( 0, math.sqrt(sigma2), size=(np.shape(possibles_d[-1])) ) )


    # BIASED ESTIMATOR
    ML_function = sum( (np.array(estimated_d) - np.array(possibles_d))**2 )
    max_index = np.argmin(ML_function) # argmin because this is not properly the ML function but a derivation

    
    max_row = int(max_index/space.get_npoint())
    max_column = max_index%space.get_npoint()
    return position_grid[max_row][max_column]

# def ML_estimation_unbias(real_position, anchors, space, p_LOS=0.5, sigma2=1, b=1):
def ML_estimation_unbias(estimated_d, possibles_d, position_grid, space, bias, sigma2=1, p_LOS=0.5):
    # positions_grid = space.get_grid()

    # real_d = []

    # possibles_d = []
    # estimated_d = []
    # bias = []
    # for anchor in anchors:
    #     # real distances
    #     real_d.append( np.full(np.shape(positions_grid), norm(real_position - anchor)) )

    #     # all possible distances --> ML estimator have to choise the most likelihood
    #     value = np.empty((), dtype=object)
    #     value[()] = anchor
    #     possibles_d.append(positions_grid - np.full(np.shape(positions_grid), value, dtype=object))

    #     i, j = 0, 0
    #     for i in range(space.get_npoint()):
    #         for j in range(space.get_npoint()):
    #             possibles_d[-1][i,j] = norm(possibles_d[-1][i,j])
    #             j += 1
    #         i += 1

    #     bias_value = np.random.binomial(1, 1-p_LOS)*b
    #     bias.append( np.full(shape=np.shape(possibles_d[-1]), fill_value=bias_value ) )
    #     estimated_d.append( real_d[-1] + bias[-1] + np.random.normal( 0, math.sqrt(sigma2), size=(np.shape(possibles_d[-1])) ) )

    # # UNBIASED ESTIMATOR
    # if sigma2 == 0:
    #     # for i in range(space.get_npoint()):
    #     #     for j in range(space.get_npoint()):
    #     #         for k in range(len(anchors)):
    #     pass
        
    # else:
    ML_function = np.prod(\
        (p_LOS)*np.exp(-np.array(((np.array(estimated_d) - np.array(possibles_d))**2)/(2*sigma2), dtype=float)) + \
        (1-p_LOS)*np.exp(-np.array(((np.array(estimated_d) - np.array(bias) - np.array(possibles_d))**2)/(2*sigma2),dtype=float)),\
            axis=0) # FORSE C'È UN PROB QUA
    max_index = np.argmax(ML_function) # argmin because this is not properly the ML function but a derivation

    
    max_row = int(max_index/space.get_npoint())
    max_column = max_index%space.get_npoint()
    return position_grid[max_row][max_column]