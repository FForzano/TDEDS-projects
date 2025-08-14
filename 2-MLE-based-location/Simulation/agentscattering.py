import numpy as np
import random
from cartesianspace import cartesianspace
from matplotlib import pyplot as plt
import tikzplotlib as tpl

if __name__ == "__main__":
    P = 5000 # P is the number of agent position considered
    L = 10 # L = 50m --> length of the side of the square
    delta = L/200 # space discetization parameter

    # The space grid is [-L/2,L/2]x[-L/2,L/2] with discretization parameter delta.
    # The four anchors are in the four corners.

    grid = cartesianspace(L, delta)
    axis_npoint = grid.get_npoint()

    #anchor1 = 

    for i in range(P):
        x = random.choice( grid.axis() )
        y = random.choice( grid.axis() )

        plt.scatter(x,y, c=['r'], marker='.', alpha=0.3)
    
    tpl.save("agent_scattering_def.tex")
    plt.show()