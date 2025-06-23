from metadrive.examples.ppo_expert.torch_expert import torch_expert
from metadrive.policy.base_policy import BasePolicy


class TorchPolicy(BasePolicy):
    def act(self, agent_id=None):
        action = torch_expert(self.control_object)
        self.action_info["action"] = action
        return action
