import numpy as np
import math

class cartesianspace:

    def __init__(self, L, delta):
        self.L = L
        self.delta = delta

        self.n_point = self.L/self.delta
        if int(self.n_point)%2 == 0:
            self.n_point = int(self.n_point)+1
        else:
            self.n_point = int(self.n_point)
    
    def axis(self):
        return (np.array( range( self.n_point ) ) - (self.n_point-1)/2) * self.delta

    def get_npoint(self):
        return self.n_point

    # def distance(self, P1, P2):
    #     '''
    #     P1 and P2 are two points in the cartesian plane.
    #     They must be a two tuple (x,y)
    #     '''

    #     return math.sqrt( ((P1[0]-P2[0])*self.delta)**2 + ((P1[1]-P2[1])*self.delta)**2 )

    def get_grid(self):
        value = np.empty((), dtype=object)
        value[()] = (0, 0)
        grid = np.full( shape=(self.n_point, self.n_point), fill_value=value, dtype=tuple)
        
        for row in range(self.n_point):
            for column in range(self.n_point):
                grid[row][column] = ((column - (self.n_point-1)/2)*self.delta, (row - (self.n_point-1)/2)*self.delta)
        
        return grid