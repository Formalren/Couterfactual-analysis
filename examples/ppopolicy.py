# author:"flt"
# data:7/29/2024 6:42 PM
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy
from IPython.display import Image

env = MetaDriveEnv(dict(map="C",
                        agent_policy=ExpertPolicy,
                        log_level=50,
                        traffic_density=0.2))
try:
    # run several episodes
    env.reset()
    for step in range(300):
        # simulation
        _, _, _, _, info = env.step([0, 3])
        env.render(mode="topdown",
                   window=False,
                   screen_record=True,
                   screen_size=(700, 870),
                   camera_position=(60, -63)
                   )
        if info["arrive_dest"]:
            break
    env.top_down_renderer.generate_gif()
finally:
    env.close()
