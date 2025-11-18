
TRAIN_CONFIG = {
    'alpha_range': np.arange(0.1, 7.0, 0.1).tolist(),
    'beta_range': np.arange(0.5, 5.0, 0.1).tolist(),
    'h_range': [0.5],
    'c_range': [25],
    'total': list(range(10, 40)),
    'travel_time': TravelTimeDist.UNIFORM,
    'param_estimator': AlphaEstimator.MAX_LIKELI,
}

TRAIN_CONFIG = {
    'alpha_range': np.arange(1.1, 7.0, 0.1).tolist(),
    'beta_range': np.arange(1.1, 5.0, 0.1).tolist(),
    'h_range': [0.5],
    'c_range': [25],
    'total': list(range(10, 40)),
    'travel_time': TravelTimeDist.UNIFORM,
    'param_estimator': AlphaEstimator.MAX_LIKELI,
}

TRAIN_CONFIG = {
    'alpha_range': np.arange(0.1, 7.0, 0.1).tolist(),
    'beta_range': np.arange(0.1, 5.0, 0.1).tolist(),
    'h_range': [0.5],
    'c_range': [25],
    'total': list(range(5, 40)),
    'travel_time': TravelTimeDist.UNIFORM,
    'param_estimator': AlphaEstimator.MAX_LIKELI,
}
