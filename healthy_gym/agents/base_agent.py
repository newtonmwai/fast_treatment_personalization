"""
Abstract class defining which methods an agent have
"""

class BaseAgent:
    def __init__(self, **kwargs):
        """
        Initialize the agent
        """
        pass

    def act(self, x, **kwargs):
        """
        Sample action from the policy and decide whether to stop interacting with the environment.

        Args:
            x: current context

        Returns:
            action (integer or float): Action sampled from the current policy
            done (boolean): True if the agent wants to terminate. Used for pure-exploration.
            info (dict): information about the policy.
        """
        raise Exception('Act not implemented')

    def update(self, x, action, reward, **kwargs):
        """
            Update the policy based on (x, action, reward)

            Args:
                x : batch of contexts
                action: batch of actions taken w.r.t. policy computed for x
                rewards: corresponding rewards
        """
