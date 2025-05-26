from src.model_approaches.base_model_approach import BaseModelApproach
from src.commons import methods

class OracleVar_n(BaseModelApproach):
    def __init__(self):
        self.model_name = f'Oracle_Var_n'

    @staticmethod
    def evaluate(alpha, beta, intervals, h, c, travel_time):
        total = len(intervals)
        N = total
        n = 0
        u_star = methods.get_u_star_binary_fast(N=N, alpha=alpha, beta=beta, h=h, c=c)
        t_now = 0
        next_event = intervals[n]
        while (travel_time < u_star) and (N > 0):
            t_now = min(u_star - travel_time, next_event)
            if t_now == next_event:
                N -= 1
                n += 1
                u_star = methods.get_u_star_binary_fast(N=N, alpha=alpha, beta=beta, h=h, c=c)
                t_now = 0
                next_event = intervals[n]
            else:
                break
        
        reach_time = sum(intervals[:n]) + max(u_star, travel_time)
        cost = methods.cal_cost(c=c, h=h, actual_time=sum(intervals), predicted_time=reach_time)
        return cost, n
    
    def predict(self, row:dict, override=False):
        if not override and self._check_keys(row):
            return [row[k] for k in self.prediction_keys()]

        return self.evaluate(alpha=row['alpha'], beta=row['beta'],
                        intervals=row['intervals'], h=row['h'], c=row['c'],
                        travel_time=row['travel_time'])

    def prediction_keys(self):
        return [f'cost_{self.model_name}', f'observed_n_{self.model_name}']

