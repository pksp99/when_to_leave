import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import statistics
from src.commons import methods
import numpy as np
import logging

from src.commons.methods import file_path
from src.rl_environments.base_env import BaseEnv
from typing import override


DEFAULT_CONFIG = {
    'alpha_range': [ (0 + i / 10) for i in range(1,10)],
    'beta_range': [round(i * 0.5, 1) for i in range(2, 6)],
    'h_range': [0.5],
    'c_range': [25],
    'total': range(10, 40), 
    'travel_time': 'uniform',
}


class Env1(BaseEnv):
    def __init__(self, step_size=0.1, config=DEFAULT_CONFIG):
        super(Env1, self).__init__(config=config)

    @override
    def _get_obs(self):
        return np.array([
            self.n,
            self.N,
            self.travel_time,
            self.cur_time,
            self.mean_n,
            self.std_n,
            self.u_star_hat,
            self.last_update
        ], dtype=np.float32)
    
    @override
    def cal_derived_data(self):
        super().cal_derived_data()
        
        # Unable to compute u* cases
        if self.alpha_hat * self.N <= 1 or \
            self.alpha_hat * self.N > 300 or \
             self.h / self.c >= 1 / self.beta or \
             self.h / self.c >= 1 / self.beta_hat:
            self.u_star_hat = self.mean_n * self.N
        else:
            self.u_star_hat = methods.get_u_star_binary_fast(self.N, self.alpha_hat, self.beta_hat, self.h, self.c)
