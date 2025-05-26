from src.model_approaches.base_model_approach import BaseModelApproach
from src.commons import methods

class OracleFix_n(BaseModelApproach):
    def __init__(self, n):
        self.n = n
        self.model_name = f'Oracle_Fix_n_{self.n}'

    @staticmethod
    def evaluate(alpha, beta, intervals, h, c, travel_time, n):
        total = len(intervals)
        N = total - n
        u_star = methods.get_u_star_binary_fast(N=N, alpha=alpha, beta=beta, h=h, c=c)
        reach_time = sum(intervals[:n]) + max(u_star, travel_time)
        cost = methods.cal_cost(c=c, h=h, actual_time=sum(intervals), predicted_time=reach_time)
        return cost
    
    def predict(self, row:dict, override=False):
        if not override and self._check_keys(row):
            return [row[k] for k in self.prediction_keys]
        
        return self.evaluate(alpha=row['alpha'], beta=row['beta'],
                        intervals=row['intervals'], h=row['h'], c=row['c'],
                        travel_time=row['travel_time'], n=self.n)
    
   
    
    def prediction_keys(self):
        return [f'cost_{self.model_name}']
    
    
 