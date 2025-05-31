from enum import Enum

class TravelTimeDist(str, Enum):
    UNIFORM = 'uniform'
    LOW = 'low'
    HIGH = 'high'

class AlphaEstimator(str, Enum):
    MOMENTS = 'method of moments'
    MAX_LIKELI = 'maximum likelihood estimation'
    MAX_LIKELI_GEN_GAMMA_BIAS = 'maximum likelihood estimation from the generalized gamma distribution biased'
    MAX_LIKELI_GEN_GAMMA_UNBIAS = 'maximum likelihood estimation from the generalized gamma distribution unbiased'




DEFAULT_CONFIG = {
    'alpha_range': [ (0 + i / 10) for i in range(1,10)],
    'beta_range': [round(i * 0.5, 1) for i in range(2, 6)],
    'h_range': [0.5],
    'c_range': [25],
    'total': range(10, 40), 
    'travel_time': TravelTimeDist.UNIFORM,
    'param_estimator': AlphaEstimator.MOMENTS,
}