import numpy as np
from .base_agent import BaseAgent
import kl

class TrackAndStop(BaseAgent):
    def __init__(self, n_arms, confidence, type='Gaussian', sigma=1, prior=None):
        """
        Track-n-stop algorithm for pure-exploration. See Kaufmann et al. https://www.jmlr.org/papers/volume17/kaufman16a/kaufman16a.pdf.
        Only for Multi-armed Bandit problems without any context.

        Args:
            n_arms (int): Number of arms in the pure exploration problem.
            confidence (float): Confidence level after which the algorithm will return a recommendation.
            type (string): The reward distribution of the arms. Can be Bernoulli or Gaussian. Default Gaussian.
            sigma (float or ndarray): Standard deviation of the reward distribution. Default sigma=1.
            prior (ndarray): Prior on the arms. A vector of mean reward which we will initialize the arms with.
            Default is None which means that the sample mean of each arm will be set to 0.
        """

        self.n_arms = n_arms
        self.confidence = confidence
        self.type = type
        self.sigma = sigma
        if prior is None:
            self.prior = np.zeros(n_arms)
        else:
            assert n_arms == prior
            self.prior = prior

        self.t = 0

    def reset(self):
        self.t = 0

    def act(self, x=None, **kwargs):
        """
        Sample an action from the policy

        Args:
            x: current context. Not used in the standard track-n-stop

        Returns:
            action (int): Index of action sampled from policy
            done (boolean): True if confidence level of arm => self.confidence
            info (dict): contains
        """


    def compute_allocation(self):
        """
        Compute the optimal allocation rule given the sample means. 
        """