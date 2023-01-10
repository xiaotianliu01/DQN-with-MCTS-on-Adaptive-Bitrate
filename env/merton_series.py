import numpy as np
import math
var = 11

class merton(object):

    def __init__(self, length):
        base = 5686944/5
        start = 0

        delta = 0.01
        delta_t = var
        u = 0.5*delta*delta
        a = 0
        b = 0.01
        lamda = var
        
        self.series_bw = []
        self.drump = []
        self.no_drump = []
        self.no_drump.append(start)
        self.series_bw.append(start)
        self.drump.append(0)
        for i in range(length):
            Z = np.random.normal(0, 1)
            N = np.random.poisson(lamda)
            Z_2 = np.random.normal(0, 2)
            M = a*N + b*(N**0.5)*Z_2
            no_drump = self.no_drump[-1] + u - 0.5*delta*delta + (delta_t**0.5)*delta*Z
            drump = M
            self.drump.append(drump)
            self.no_drump.append(no_drump)
            new_X = self.series_bw[-1] + u - 0.5*delta*delta + (delta_t**0.5)*delta*Z + M
            self.series_bw.append(new_X)
        
        self.series_bw = [math.exp(i)*base for i in self.series_bw]
        self.drump = [math.exp(i)*base for i in self.drump]
        self.no_drump = [math.exp(i)*base for i in self.no_drump]

    def regenrate_bw(self, last, length):

        new_sequence = []
        for i in range(length):
            delta = 0.01
            delta_t = var
            u = 0.5*delta*delta
            a = 0
            b = 0.01
            lamda = var
            Z = np.random.normal(0, 1)
            N = np.random.poisson(lamda)
            Z_2 = np.random.normal(0, 2)
            M = a*N + b*(N**0.5)*Z_2
            new_X = u - 0.5*delta*delta + (delta_t**0.5)*delta*Z + M
            new_sequence.append(last*math.exp(new_X))
            last = new_sequence[-1]
        
        return new_sequence
    
    def __getitem__(self, key = 0):
        return self.series_bw[key]
