"""
The existing layer names and shapes in numpy file:
(note that the terms with "value" in it are removed to save space).
default_policy/fc_1/kernel (275, 256)
default_policy/fc_1/bias (256,)
default_policy/fc_value_1/kernel (275, 256)
default_policy/fc_value_1/bias (256,)
default_policy/fc_2/kernel (256, 256)
default_policy/fc_2/bias (256,)
default_policy/fc_value_2/kernel (256, 256)
default_policy/fc_value_2/bias (256,)
default_policy/fc_out/kernel (256, 4)
default_policy/fc_out/bias (4,)
default_policy/value_out/kernel (256, 1)
default_policy/value_out/bias (1,)
"""

import os.path as osp
from metadrive.engine.logger import get_logger
import numpy as np
import random

from metadrive.engine.engine_utils import get_global_config
from metadrive.obs.state_obs import LidarStateObservation

ckpt_path = osp.join(osp.dirname(__file__), "EGPO.npz")
_expert_weights = None
_expert_observation = None

logger = get_logger()


def obs_correction(obs):
    # due to coordinate correction, this observation should be reversed
    obs[15] = 1 - obs[15]
    obs[10] = 1 - obs[10]
    return obs


def egpo(vehicle, deterministic=False, need_obs=False):
    global _expert_weights
    global _expert_observation
    expert_obs_cfg = dict(
        lidar=dict(num_lasers=240, distance=50, num_others=4, gaussian_noise=0.0, dropout_prob=0.0),
        side_detector=dict(num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
        lane_line_detector=dict(num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0),
        random_agent_model=False,
        NORMAL_SPEED=random.randint(100, 180)
    )
    origin_obs_cfg = vehicle.config.copy()
    # TODO: some setting in origin cfg will not be covered, then they may change the obs shape

    if _expert_weights is None:
        _expert_weights = np.load(ckpt_path)
        config = get_global_config().copy()
        config["vehicle_config"].update(expert_obs_cfg)
        _expert_observation = LidarStateObservation(config)
        assert _expert_observation.observation_space.shape[0] == 275, "Observation not match"
        logger.info("Torch is not available. Use numpy PPO expert.")

    vehicle.config.update(expert_obs_cfg)
    obs = _expert_observation.observe(vehicle)
    vehicle.config.update(origin_obs_cfg)
    obs = obs_correction(obs)
    weights = _expert_weights
    obs = obs.reshape(1, -1)
    x = np.matmul(obs, weights["default_policy/fc_1/kernel"]) + weights["default_policy/fc_1/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_2/kernel"]) + weights["default_policy/fc_2/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_out/kernel"]) + weights["default_policy/fc_out/bias"]
    x = x.reshape(-1)
    mean, log_std = np.split(x, 2)
    if deterministic:
        return (mean, obs) if need_obs else mean
    std = np.exp(log_std)
    action = np.random.normal(mean, std)
    ret = action
    # ret = np.clip(ret, -1.0, 1.0) all clip should be implemented in env!
    return (ret, obs) if need_obs else ret


def load_weights(path: str):
    """
    Load NN weights
    :param path: weights file path path
    :return: NN weights object
    """
    try:
        model = np.load(path)
        return model
    except FileNotFoundError:
        print("Can not find {}, didn't load anything".format(path))
        return None


def value(obs, weights):
    """
    Given weights, return the evaluation to one state/obseration
    :param obs: observation
    :param weights: variable weights of NN
    :return: value
    """
    if weights is None:
        return 0
    obs = obs.reshape(1, -1)
    x = np.matmul(obs, weights["default_policy/fc_value_1/kernel"]) + weights["default_policy/fc_value_1/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_value_2/kernel"]) + weights["default_policy/fc_value_2/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/value_out/kernel"]) + weights["default_policy/value_out/bias"]
    ret = x.reshape(-1)
    return ret


# if __name__ == '__main__':
#     for i in range(100):
#         print("Weights? ", type(_expert_weights))
#         ret = expert(np.clip(np.random.normal(0.5, 1, size=(275,)), 0.0, 1.0))
#         print("Return: ", ret)
