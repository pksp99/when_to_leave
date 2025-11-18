import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import statistics
from src.commons import methods
import numpy as np
import logging

from src.commons.methods import file_path
from src.rl_environments.base_env_v2 import BaseEnvV2
from src.commons.constants import DEFAULT_CONFIG
from typing import override


class Env1V2(BaseEnvV2):
    def __init__(self, config=DEFAULT_CONFIG):
        super(Env1V2, self).__init__(config=config)

    @override
    def _get_obs(self):
        return np.array([
            self.n,
            self.N,
            self.travel_time,
            self.mean_n,
            self.std_n,
            self.u_star_hat,
        ], dtype=np.float32)
    
    @override
    def cal_derived_data(self):
        super().cal_derived_data()
          # Unable to compute u* cases
        if self.alpha_hat * self.N <= 1 or \
            self.alpha_hat * self.N > 300 or \
             self.h / self.c >= 1 / self.beta_hat:
            self.u_star_hat = self.mean_n * self.N
        else:
            self.u_star_hat = methods.get_u_star_binary_fast(self.N, self.alpha_hat, self.beta_hat, self.h, self.c)
            
    @override
    def step(self, action):

        if self.n >= self.total:
            action = 1

        if action >= 1:
            cur_time = self.cum_sum_intervals[self.n - 1]
            reach_time = cur_time + max(self.u_star_hat, self.travel_time)
            cost = methods.cal_cost(c=self.c, h=self.h, actual_time=self.cum_sum_intervals[-1],
                                    predicted_time=reach_time)
            self.obs_intervals = self.intervals[:self.total]
            self.final_observed_n = self.n
            self.n = self.total
            self.N = 0
            return self._get_obs(), -cost, True, False, self._get_info()
        else:
            self.n += 1
            self.cal_derived_data()
            return self._get_obs(), 0, False, False, self._get_info()
