import random
import numpy as np
from scipy.stats import norm

# 引入噪声
class OU:

    def __init__(self, processes, mu = 0, theta=0.1, sigma=3):
        self.dt = 1e-2
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.processes = processes
        self.state = np.ones(self.processes) * self.mu

    def reset(self):
        self.state = np.ones(self.processes) * self.mu

    # OU噪声
    def evolve(self):
        X = self.state
        # dw = norm.rvs(scale=self.dt, size=self.processes)
        # dx = self.theta * (self.mu - X) * self.dt + self.sigma * dw
        dx = self.theta * (self.mu - X) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.processes)
        self.state = X + dx
        return self.state
