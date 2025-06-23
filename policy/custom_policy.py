from metadrive.examples.ppo_expert.custom_expert import rule_expert
from metadrive.policy.base_policy import BasePolicy


class CumtomPolicy(BasePolicy):
    def act(self, agent_id=None):
        action = rule_expert(self.control_object)
        self.action_info["action"] = action
        return action
