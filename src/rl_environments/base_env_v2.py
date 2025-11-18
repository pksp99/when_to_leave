import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import statistics
from src.commons import methods, generate_data
import numpy as np
from typing import override
from src.commons.constants import DEFAULT_CONFIG


class BaseEnvV2(gym.Env):
    def __init__(self, config=DEFAULT_CONFIG):
        super(BaseEnvV2, self).__init__()
        self.config = config
        self.reset()

        # 0 = wait, others are leaave
        self.action_space = Discrete(5)

        self.observation_space = Box(low=0, high=np.inf, shape=(len(self._get_obs()),), dtype=np.float32)

    def _get_obs(self):
        return np.array([
            self.n,
            self.N,
            self.travel_time,
            self.mean_n,
            self.std_n,
        ], dtype=np.float32)

    def _get_info(self):
        class_vars = {k: v for k, v in vars(self).items() if not k.startswith('__') and not callable(v)}

        return class_vars

    def cal_derived_data(self):
        self.N = self.total - self.n
        self.obs_intervals = self.intervals[:self.n]

        self.mean_n = statistics.mean(self.obs_intervals)
        self.std_n = statistics.stdev(self.obs_intervals)

        self.alpha_hat, self.beta_hat = methods.gamma_estimate_parameters(self.n, self.intervals, param_estimator=self.config['param_estimator'])

    def reset(self, seed=None, options=None, config=None, row=None):

        if row is None:
            row = generate_data.get_realized_data(self.config)

        self.alpha = row['alpha'] if 'alpha' in row else None
        self.beta = row['beta'] if 'beta' in row else None

        self.h = row['h']
        self.c = row['c']
        self.total = int(row['total'])
        self.travel_time = row['travel_time']
        if isinstance(row['intervals'], list):
            self.intervals = row['intervals']
        else:
            self.intervals = row['intervals'].tolist()

        self.cum_sum_intervals = np.cumsum(self.intervals)

        self.n = 3

        self.cal_derived_data()

        return self._get_obs(), self._get_info()

    def step(self, action):

        if self.n >= self.total:
            action = 1

        if action >= 1:
            cur_time = self.cum_sum_intervals[self.n - 1]
            cost = methods.cal_cost(c=self.c, h=self.h, actual_time=self.cum_sum_intervals[-1],
                                    predicted_time=cur_time + self.travel_time)
            self.obs_intervals = self.intervals[:self.total]
            self.final_observed_n = self.n
            self.n = self.total
            self.N = 0
            return self._get_obs(), -cost, True, False, self._get_info()
        else:
            self.n += 1
            self.cal_derived_data()
            return self._get_obs(), 0, False, False, self._get_info()

    def render(self, mode='human'):
        print(self._get_obs())
