import numpy as np
import os
import pandas as pd

from src.commons import methods
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from src.commons import methods
import json


def get_realized_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a single row of synthetic data based on the given configuration."""
    alpha = np.random.choice(config['alpha_range'])
    beta = np.random.choice(config['beta_range'])
    h = np.random.choice(config['h_range'])
    c = np.random.choice(config['c_range'])
    total = np.random.choice(config['total'])

    intervals = np.random.gamma(shape=alpha, scale=beta, size=total)

    if config['travel_time'] == 'high':
        travel_time = np.sum(intervals[3:]) - np.random.gamma(shape=2, scale=alpha * beta)
    elif config['travel_time'] == 'low':
        travel_time = np.random.gamma(shape=2, scale=2 * alpha * beta)
    else:  # 'uniform'
        travel_time = np.sum(intervals[3:]) * np.random.uniform(0, 1)

    travel_time = max(alpha * beta, travel_time)

    return {
        'alpha': alpha,
        'beta': beta,
        'h': h,
        'c': c,
        'total': total,
        'intervals': intervals,
        'travel_time': travel_time,
    }


def generate(config: Dict[str, Any], row_count: int) -> str:
    """Generate and persist a synthetic dataset based on the given configuration."""
    dataset_name = f"{methods.get_config_hash(config)}_{methods.abbreviate_number(row_count)}"
    data_file_path = Path(methods.file_path(dataset_name, 'data'))

    pickle_path = data_file_path.with_suffix('.pkl')

    if pickle_path.is_file():
        print(f"[INFO] Dataset already exists:\n{pickle_path}\n")
        return str(pickle_path)

    print(f"[INFO] Generating new dataset:\n{data_file_path}\n")

    # Efficiently generate all data records
    records = [get_realized_data(config) for _ in range(row_count)]
    df = pd.DataFrame(records)

    # Persist the dataset
    df.to_csv(data_file_path.with_suffix('.csv'), index=False)
    df.to_pickle(pickle_path)
    with open(str(data_file_path.with_suffix('.json')), 'w') as f:
        json.dump(config, f, indent=4)

    return str(pickle_path)