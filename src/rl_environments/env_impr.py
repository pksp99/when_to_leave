import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import statistics
from src.commons import methods
import numpy as np
import logging

from src.rl_environments.env1 import Env1
from typing import override

DEFAULT_CONFIG = {
    'alpha_range': [ (0 + i / 10) for i in range(1,10)],
    'beta_range': [round(i * 0.5, 1) for i in range(2, 6)],
    'h_range': [0.5],
    'c_range': [25],
    'total': range(10, 40), 
    'travel_time': 'uniform',
}

class EnvImpr(Env1):
    def __init__(self, config=DEFAULT_CONFIG, step_size=0.1):
        super(EnvImpr, self).__init__(config=config)

    @override
    def _get_obs(self):
        return np.array([
            self.remain_ratio,
            self.scaled_diff_u_tt,
            self.scaled_diff_last_update
        ], dtype=np.float32)
    
    @override
    def cal_derived_data(self):
        super().cal_derived_data()
        self.remain_ratio = self.N / (self.total)
        self.scaled_diff_u_tt = (self.u_star_hat - self.travel_time) / self.mean_n
        self.scaled_diff_last_update = (self.cur_time - self.last_update) / self.mean_n