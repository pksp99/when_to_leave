import statistics
from src.model_approaches.base_model_approach import BaseModelApproach
from src.commons import methods
from src.commons.constants import AlphaEstimator

class PTO_Fix_n(BaseModelApproach):
    def __init__(self, n, config):
        self.n = n
        self.model_name = f'PTO_Fix_n_{self.n}'
        self.config = config

    @staticmethod
    def evaluate(intervals, h, c, travel_time, n, param_estimator:AlphaEstimator):
        total = len(intervals)
        N = total - n
        mean_n = statistics.mean(intervals[:n])
        alpha_hat, beta_hat = methods.gamma_estimate_parameters(n=n, intervals=intervals, param_estimator=param_estimator)
        u_star_hat = methods.robust_u_star_estimator(N=N, alpha=alpha_hat, beta=beta_hat, h=h, c=c, mean_n=mean_n)
        reach_time = sum(intervals[:n]) + max(u_star_hat, travel_time)
        cost = methods.cal_cost(c=c, h=h, actual_time=sum(intervals), predicted_time=reach_time)
        return cost

    def predict(self, row: dict, override=False):
        if not override and self._check_keys(row):
            return [row[k] for k in self.prediction_keys()]

        return self.evaluate(intervals=row['intervals'], h=row['h'], c=row['c'],
                             travel_time=row['travel_time'], n=self.n, param_estimator=self.config['param_estimator'])

    def prediction_keys(self):
        return [f'cost_{self.model_name}']