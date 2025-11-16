import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import statistics
from src.commons import methods
import numpy as np
import logging

from src.rl_environments.env1_v2 import Env1V2
from src.commons.constants import DEFAULT_CONFIG
from typing import override


class EnvImpr2V2(Env1V2):
    def __init__(self, config=DEFAULT_CONFIG):
        super(EnvImpr2V2, self).__init__(config=config)

    @override
    def _get_obs(self):
        return np.array([
            self.remain_ratio,
            self.scaled_diff_u_tt_alpha_hat,
            self.scaled_diff_u_tt_beta_hat,
        ], dtype=np.float32)
    
    @override
    def cal_derived_data(self):
        super().cal_derived_data()
        self.remain_ratio = self.N / (self.total)
        self.scaled_diff_u_tt_alpha_hat = (self.u_star_hat - self.travel_time) / self.alpha_hat
        self.scaled_diff_u_tt_beta_hat = (self.u_star_hat - self.travel_time) / self.beta_hat
        