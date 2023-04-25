import numpy as np
from copy import deepcopy

PRECISION = 1e-24

class BayesDesigner:
    def __init__(self, models, belief, K=8, delta=0.05, n_draws=100):
        """
        Initialize the agent
        """
        self.models = models
        self.belief = belief
        self.delta = delta
        self.Z_tau = 0
        self.is_done = False
        self.K = K
        self.N = np.zeros(self.K)
        self.t = 0
        self.latent_dim = len(models)
        self.n_draws = n_draws # number of draws for each arm used in estimating the utility

    def act(self, x, **kwargs):
        """
        Play action that maximizes the expected information gain

        Args:
            x: current context

        Returns:
            action (integer or float): Action sampled from the current policy
            done (boolean): True if the agent wants to terminate. Used for pure-exploration.
            info (dict): information about the policy.
        """

        # get prior
        p_z = self.belief.prior
        p_a = self.compute_p_a(p_z, x)
        utility = self.compute_utility(x, p_a)

        best_arm = np.argmax(p_a)
        if p_a[best_arm] > 1 - self.delta:
            a_t = best_arm
            is_done = True
        else:
            a_t = np.argmax(utility)
            is_done = False

        self.N[a_t] += 1
        info = {}

        info['recommended_a_tau'] = None

        zz = kwargs.get('z', None)

        info['Posterior'] = p_z
        info['Model'] = np.argmax(p_z)
        info['N'] = self.N
        info['recommended_a_tau'] = a_t

        self.is_done = is_done

        return a_t, is_done, info

    def update(self, x, action, reward, **kwargs):
        """
            Update the policy based on (x, action, reward)

            Args:
                x : batch of contexts
                action: batch of actions taken w.r.t. policy computed for x
                rewards: corresponding rewards
        """

        """
        Update beliefs. Takes batch data

        Args:
            x: batch of contexts
            action: batch of actions
            reward: batch of rewards
        """
        if not isinstance(x, list):
            x = [x]
            action = [action]
            reward = [reward]

        for x_i, a_i, r_i in zip(x, action, reward):
            self.belief.update(x_i, a_i, r_i)

    def compute_utility(self, x, p_a):
        utility = []
        for a in range(self.K):
            samples = self.belief.sample_reward(x, a, n_draws=self.n_draws)
            gain = [self.utility(r, x, a, p_a) for r in samples]
            utility.append(np.mean(gain))
        return np.array(utility)



    def kl_div(self, p, q):
        return np.sum( p * np.log((p + PRECISION) / (q + PRECISION)))

    def utility(self, r, x, a, p_a):
        posterior_z = self.belief.pseudo_update(x, a, r)
        posterior_a = self.compute_p_a(posterior_z, x)
        return self.kl_div(posterior_a, p_a)

    def compute_p_a(self, posterior, x):
        """
        Compute P(A^* = a | H_t)
        """
        means = [model.predict(x) for model in self.models]
        p_a = np.zeros(self.K)
        for z in range(self.latent_dim):
            a_star = np.argmax(means[z])
            p_a[a_star] += posterior[z]

        return p_a

