import concurrent.futures
import logging
import os
import random
import statistics

import numpy as np
import pandas as pd
from tqdm import tqdm

import methods
from methods import file_path


def configure_logging(log_file):
    logging.basicConfig(
        filename=file_path(log_file, 'data'),
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def process_iter(config, log_file, index, row=None):
    # Configure logging
    configure_logging(log_file)

    if row is None:
        # Randomly select parameters from config ranges
        row = {key: random.choice(config[key + '_range']) for key in ['alpha', 'beta', 'h', 'c', 'N', 'n']}
        alpha, beta, h, c, N, n = row.values()
        intervals = np.random.gamma(shape=alpha, scale=beta, size=N + n)
    else:
        # Use existing parameters
        alpha, beta, h, c, N, n = row['alpha'], row['beta'], row['h'], row['c'], row['N'], row['n']
        intervals = [float(x) for x in row['intervals_str'].split('_')]

    # Generate intervals and calculate statistics
    mean_n = statistics.mean(intervals[:n])
    std_n = statistics.stdev(intervals[:n])
    alpha_hat, beta_hat = methods.gamma_estimate_parameters(n, intervals)

    row.update({
        'mean_n': mean_n,
        'std_n': std_n,
        'alpha_hat': alpha_hat,
        'beta_hat': beta_hat,
        'intervals_str': '_'.join(map(str, intervals))
    })

    logging.info(f"Start process-{index}: {row}")

    # Unable to compute u* cases
    if alpha_hat * N <= 1 or alpha_hat * N > 600:
        logging.critical(f'End process-{index}: alpha_hat < 1')
        return None
    elif h / c >= 1 / beta:
        logging.critical(f'End process-{index}: Impossible beta: {beta}, for h: {h} and c: {c}')
        return None
    elif h / c >= 1 / beta_hat:
        logging.critical(f'End process-{index}: Impossible beta_hat: {beta_hat}, for h: {h} and c: {c}')
        return None

    # Compute additional metrics
    u = methods.cal_actual_time(n, intervals)
    u_star = methods.get_u_star_binary_fast(N, alpha, beta, h, c)
    u_star_hat = methods.get_u_star_binary_fast(N, alpha_hat, beta_hat, h, c)
    z = u_star / u_star_hat
    optimal_cost = methods.cal_cost(c, h, u, u_star)
    actual_cost = methods.cal_cost(c, h, u, u_star_hat)

    row.update({
        'u': u,
        'u_star': u_star,
        'u_star_hat': u_star_hat,
        'z': z,
        'optimal_cost': optimal_cost,
        'actual_cost': actual_cost
    })

    logging.info(f'End process-{index}: {row}')
    return row


def update_progress(_):
    global pbar
    pbar.update()


def generate(config, output: str, log_file: str, n: int = 1000, df: pd.DataFrame | None = None, overwrite: bool = True):
    # Delete log file if exists
    if os.path.exists(file_path(log_file, 'data')):
        os.remove(file_path(log_file, 'data'))

    if overwrite == False and os.path.exists(file_path(output, 'data')):
        return

    global pbar
    results = []
    if df is None:
        pbar = tqdm(total=n, desc=f"Processing {output}")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            submits = [executor.submit(process_iter, config, log_file, i) for i in range(n)]
            for f in concurrent.futures.as_completed(submits):
                update_progress(f.result())
                results.append(f.result())
    else:
        pbar = tqdm(total=len(df), desc=f"Processing {output}")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            submits = [executor.submit(process_iter, config, log_file, i, row.to_dict()) for i, row in df.iterrows()]
            for f in concurrent.futures.as_completed(submits):
                update_progress(f.result())
                results.append(f.result())

    pbar.close()
    results = [x for x in results if x is not None]
    df = pd.DataFrame(results)
    df.to_csv(file_path(output, 'data'), index=False)
