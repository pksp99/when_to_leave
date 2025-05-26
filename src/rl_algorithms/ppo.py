from src.commons import methods
from pathlib import Path
import stable_baselines3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import pickle
import subprocess
import sys
import os

def make_env(env_lambda, rank):
    def _init():
        return env_lambda()
    return _init

class PPO():
    def __init__(self, Env_class, train_config, total_timestamps):
        self.config = train_config
        self.Env_class = Env_class
        self.total_timestamps = total_timestamps
        self.model_name = f"{type(self).__name__}_{Env_class.__name__}_{methods.get_config_hash(train_config)}_{methods.abbreviate_number(total_timestamps)}"
        self.model = self.train()
        self.env = Env_class(train_config)

    def predict(self, row):
        state, _ = self.env.reset(row=row)
        done = False
        total_reward = 0
        while not done:
            action, _ = self.model.predict(state)  
            state, reward, done, _, info = self.env.step(action)
            total_reward += reward
        cost = -total_reward
        u_rl = self.env.cur_time + self.env.travel_time
        observed_n = self.env.final_observed_n
        return cost, observed_n, u_rl

    def train(self):
        model_file_path = Path(methods.file_path(self.model_name, 'models')).with_suffix('.zip')

        if model_file_path.is_file():
            print(f"[INFO] PPO Model already trained:\n{model_file_path}\n")
        
        else:
            print(f"[INFO] Training PPO Model:\n{model_file_path}\n")

            temp_dict = {
                'Env_class': self.Env_class,
                'config': self.config,
                'total_timestamps': self.total_timestamps,
                'model_file_path': model_file_path
            }
            with open(model_file_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(temp_dict, f)
            
            subprocess.run([sys.executable, os.path.abspath(__file__)] + [str(model_file_path.with_suffix('.pkl'))])

            os.remove(model_file_path.with_suffix('.pkl'))

        return stable_baselines3.PPO.load(model_file_path)


        
    


def train(Env_class, config, total_timestamps, model_file_path):
    n_cpus = os.cpu_count()

    print(f"Number of processors: {n_cpus}")
    # env = SubprocVecEnv([make_env(lambda: Env_class(config), i) for i in range(n_cpus)])
    env = make_vec_env(lambda: Env_class(config=config), n_envs=n_cpus, vec_env_cls=SubprocVecEnv)

    model = stable_baselines3.PPO("MlpPolicy", env, verbose=1, device='cpu')

    print(model.n_envs)

    print(f"Training model for {total_timestamps} timesteps")
    # Start training
    model.learn(total_timesteps=total_timestamps)

    model.save(model_file_path)

    return model

if __name__ == '__main__':
    try:
        temp_pickle_path = Path(sys.argv[1])
    except:
        temp_pickle_path = 'tp.pkl'
        pass
    if os.path.exists(temp_pickle_path):
        with open(temp_pickle_path, 'rb') as f:
            loaded_arguments = pickle.load(f)
        train(**loaded_arguments)

    else:
        TRAIN_CONFIG = {
            'alpha_range': [ (0 + i / 10) for i in range(1,10)],
            'beta_range': [round(i * 0.5, 1) for i in range(2, 6)],
            'h_range': [0.5],
            'c_range': [25],
            'total': range(10, 40), 
            'travel_time': 'uniform',
        }

        from src.rl_environments.env_impr import EnvImpr as Env
        p = PPO(Env, train_config=TRAIN_CONFIG, total_timestamps=5000)