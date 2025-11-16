from pathlib import Path
import os
import sys
import pickle
import json
import subprocess
from typing import Any, Callable, Tuple

from stable_baselines3 import PPO as SB3PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.commons import methods
from src.model_approaches.base_model_approach import BaseModelApproach


def make_env(env_fn: Callable[[], Any], rank: int) -> Callable[[], Any]:
    """Wraps an env constructor for multiprocessing."""
    def _init():
        return env_fn()
    return _init


class PPO(BaseModelApproach):
    def __init__(self, EnvClass: type, config: dict, timesteps: int):
        self.EnvClass = EnvClass
        self.config = config
        self.timesteps = timesteps

        self.model_name = self._compose_model_name()
        self.model_path = Path(methods.file_path(self.model_name, 'models')).with_suffix('.zip')
        self.env = EnvClass(config)
        self.model = self._load_or_train()

    def _compose_model_name(self) -> str:
        hash_id = methods.get_config_hash(self.config)
        abbrev_steps = methods.abbreviate_number(self.timesteps)
        return f"{type(self).__name__}_{self.EnvClass.__name__}_{hash_id}_{abbrev_steps}"

    def _load_or_train(self):
        if self.model_path.exists():
            print(f"[INFO] Loaded trained PPO model from:\n{self.model_path}")
        else:
            print(f"[INFO] Training PPO model...\nSaving to: {self.model_path}")
            self._train_via_subprocess()
        return SB3PPO.load(self.model_path)

    def _train_via_subprocess(self):
        temp_pickle = self.model_path.with_suffix('.pkl')
        with open(temp_pickle, 'wb') as f:
            pickle.dump({
                'EnvClass': self.EnvClass,
                'config': self.config,
                'timesteps': self.timesteps,
                'model_path': self.model_path
            }, f)

        subprocess.run([sys.executable, __file__, str(temp_pickle)], check=True)
        temp_pickle.unlink()


    def predict(self, row: Any, override=False) -> Tuple[float, int]:
        if not override and self._check_keys(row):
            return [row[k] for k in self.prediction_keys()]
        
        state, _ = self.env.reset(row=row)
        done = False
        total_reward = 0

        while not done:
            action, _ = self.model.predict(state)
            state, reward, done, _, _ = self.env.step(action)
            total_reward += reward

        cost = -total_reward
        observed_n = self.env.final_observed_n
        return cost, observed_n

    def prediction_keys(self) -> list[str]:
        return [f'cost_{self.model_name}', f'observed_n_{self.model_name}']


def train(EnvClass: type, config: dict, timesteps: int, model_path: Path):
    num_envs = os.cpu_count() or 1
    print(f"[INFO] Launching training with {num_envs} parallel environments")

    env = make_vec_env(lambda: EnvClass(config=config), n_envs=num_envs, vec_env_cls=SubprocVecEnv)
    model = SB3PPO("MlpPolicy", env, verbose=1, device='cpu')

    print(f"[INFO] Training PPO for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(model_path)
    with open(str(model_path.with_suffix('.json')), 'w') as f:
        json.dump(config, f, indent=4)
    print(f"[INFO] Model saved to {model_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[ERROR] Missing pickle path argument.")
        sys.exit(1)

    pickle_path = Path(sys.argv[1])
    if not pickle_path.exists():
        print(f"[ERROR] Pickle file does not exist: {pickle_path}")
        sys.exit(1)

    with open(pickle_path, 'rb') as f:
        args = pickle.load(f)

    train(**args)
