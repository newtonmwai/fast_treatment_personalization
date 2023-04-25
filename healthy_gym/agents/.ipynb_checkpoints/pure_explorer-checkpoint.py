import numpy as np
from copy import deepcopy


class DivergenceExplorer:
    def __init__(self, models, belief, K, confidence=0.9):
        """
        Initialize the agent
        """
        self.models = models
        self.K = K
        self.latent_dim = len(models)
        self.belief = belief
        self.confidence = confidence

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

        # Pick "current" model
        z_t = self.belief.sample()
        # Current belief and kl matrix
        posterior = deepcopy(self.belief.prior)  # size = [Z]
        kl_matrix = self.belief.kl_matrix(x)[z_t, :, :]  # size = [Z, A]
        # Find best arm
        w = posterior.reshape(len(posterior), 1) * kl_matrix  # size = [Z, A]
        kl_sum = np.sum(w, axis=0)  # size = [A]
        action = np.argmax(kl_sum)

        recommendation = np.argmax(posterior)
        is_done = posterior[recommendation] > self.confidence
        info = {'Posterior': posterior, 'Model': recommendation}

        p_a = self.compute_p_a(posterior, x)
        leader_arm = np.argmax(p_a)

        is_done = p_a[leader_arm] > self.confidence
        if is_done:
            action = leader_arm

        return action, is_done, info

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

class TopTwoThompsonSampling:

    def __init__(self, models, belief, K, confidence=0.9, theta=0.5):
        """
        Standard Thompson Sampling (TS) for the Multi-armed Bandit assuming no structure between the actions

        Args:
            models: Set of possible models
            belief (ModelBelief): Belief over the models
            confidence: At which confidence level the algorithm should terminate in the pure-exploration setting.
                If None, the algorithm will never terminate.
            theta: Controls the rejection sampling in act.
        """

        #self.k = len(models)
        self.models = models
        self.belief = belief
        self.K = K
        self.latent_dim = len(models)
        self.confidence = confidence
        self.theta = theta

    def act(self, x, **kwargs):
        """
        Sample an action from the TS policy

        Args:
            x: current context, used in the contextual version of TS.

        Returns:
            action (int): Index of action sampled from policy
            done (boolean): True if confidence level of arm => self.confidence. False if self.confidence == None
            info (dict): contains
        """

        posterior = deepcopy(self.belief.prior)
        leader = np.argmax(posterior)
        model = self.belief.sample()
        p_a = self.compute_p_a(posterior, x)

        if np.random.uniform() < self.theta:
            idx = np.random.choice(
                len(self.models), p=posterior, size=2, replace=False)
            if idx[0] != leader:
                model = idx[0]
            else:
                model = idx[1]

            # while model == leader:
            #    model = self.belief.sample()

        y = self.models[model].predict(x)
        action = np.argmax(y)

        info = {'Posterior': posterior, 'Model': leader}
        leader_arm = np.argmax(p_a)

        is_done = p_a[leader_arm] > self.confidence
        if is_done:
            action = leader_arm

        return action, is_done, info

    def update(self, x, action, reward, **kwargs):
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
