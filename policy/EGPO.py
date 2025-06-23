from metadrive.examples.ppo_expert.EGPO import egpo
from metadrive.policy.base_policy import BasePolicy


class EGPO(BasePolicy):
    def act(self, agent_id=None):
        action = egpo(self.control_object)
        self.action_info["action"] = action
        return action
