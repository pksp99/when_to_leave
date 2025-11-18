import statistics
import numpy as np
from src.model_approaches.base_model_approach import BaseModelApproach
from src.commons import methods
from src.commons.constants import AlphaEstimator

class PTO_Var_n_sp(BaseModelApproach):
    def __init__(self, param_estimator: AlphaEstimator):
        self.model_name = f'PTO_Var_n_sp'
        self.param_estimator = param_estimator

    @staticmethod
    def evaluate(intervals, h, c, travel_time, param_estimator:AlphaEstimator):
        total = len(intervals)
        N = total - 3
        n = 3
        mean_n = statistics.mean(intervals[:n])
        alpha_hat, beta_hat = methods.gamma_estimate_parameters(n=n, intervals=intervals, param_estimator=param_estimator)
        u_star_hat = methods.robust_u_star_estimator(N=N, alpha=alpha_hat, beta=beta_hat, h=h, c=c, mean_n=mean_n)
        t_now = 0
        next_event = np.random.gamma(alpha_hat, beta_hat)
        while (travel_time < u_star_hat) and (N > 0):
            t_now = min(u_star_hat - travel_time, next_event)
            if t_now == next_event:
                N -= 1
                n += 1
                mean_n = statistics.mean(intervals[:n])
                alpha_hat, beta_hat = methods.gamma_estimate_parameters(n=n, intervals=intervals, param_estimator=param_estimator)
                u_star_hat = methods.robust_u_star_estimator(N=N, alpha=alpha_hat, beta=beta_hat, h=h, c=c, mean_n=mean_n)
                t_now = 0
                next_event = np.random.gamma(alpha_hat, beta_hat)
            else:
                break
        
        reach_time = sum(intervals[:n]) + travel_time
        cost = methods.cal_cost(c=c, h=h, actual_time=sum(intervals), predicted_time=reach_time)
        return cost, n

    def predict(self, row: dict, override=False):
        if not override and self._check_keys(row):
            return [row[k] for k in self.prediction_keys()]

        return self.evaluate(intervals=row['intervals'], h=row['h'], c=row['c'], travel_time=row['travel_time'], param_estimator=self.param_estimator)

    def prediction_keys(self):
        return [f'cost_{self.model_name}', f'observed_n_{self.model_name}']