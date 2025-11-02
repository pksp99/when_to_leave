import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import statistics
from src.commons import methods
import numpy as np
import logging

from src.rl_environments.env_impr import EnvImpr
from src.commons.constants import DEFAULT_CONFIG
from typing import override


class EnvImprAB(EnvImpr):
    def __init__(self, config=DEFAULT_CONFIG, step_size=0.1):
        super(EnvImprAB, self).__init__(config=config)

    @override
    def _get_obs(self):
        return np.array([
            self.remain_ratio,
            self.scaled_diff_u_tt,
            self.scaled_diff_last_update,
            self.alpha_hat,
            self.beta_hat
        ], dtype=np.float32)
    
    @override
    def cal_derived_data(self):
        super().cal_derived_data()