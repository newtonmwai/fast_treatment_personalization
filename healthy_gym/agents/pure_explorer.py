import numpy as np
from copy import deepcopy
PRECISION = 1e-24


class DivergenceExplorer:
    def __init__(self, models, belief, K=8, confidence=0.9):
        """
        Initialize the agent
        """
        self.means = None
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
        # print("Divergence Prior: ", self.belief.prior)
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

        # print("Z_tau Div: ", p_a[leader_arm])
        is_done = p_a[leader_arm] > self.confidence
        if is_done:
            action = leader_arm

        # print("Divergence action: ", action)
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
        means = self.means  # [model.predict(x)[0] for model in self.models]
        p_a = np.zeros(self.K)
        for z in range(self.latent_dim):
            a_star = np.argmax(means[z])
            p_a[a_star] += posterior[z]

        return p_a


class TopTwoThompsonSampling:

    def __init__(self, models, belief, K=8, confidence=0.9, theta=0.5):
        """
        Standard Thompson Sampling (TS) for the Multi-armed Bandit assuming no structure between the actions

        Args:
            models: Set of possible models
            belief (ModelBelief): Belief over the models
            confidence: At which confidence level the algorithm should terminate in the pure-exploration setting.
                If None, the algorithm will never terminate.
            theta: Controls the rejection sampling in act.
        """

        # self.k = len(models)
        self.means = None
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

        y = self.means[model]
        action = np.argmax(y)

        info = {'Posterior': posterior, 'Model': leader}
        leader_arm = np.argmax(p_a)

        # print("Z_tau TTTS: ", p_a[leader_arm])
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
        means = self.means  # [model.predict(x)[0] for model in self.models]
        p_a = np.zeros(self.K)
        for z in range(self.latent_dim):
            a_star = np.argmax(means[z])
            p_a[a_star] += posterior[z]

        return p_a


class VanillaTTTSExplorer:
    def __init__(self, beliefs, confidence=None, K=8, beta=0.5):
        """
        Standard Top Two Thompson Sampling (TTTS) for the Multi-armed Bandit assuming no structure between the actions

        Args:
            beliefs: A set of beliefs, one for each action.
            explore: Whether to do pure exploration or regret minimization.

        """
        super(VanillaTTTSExplorer, self).__init__()
        self.beliefs = beliefs
        self.confidence = confidence
        self.beta = beta
        self.means = None
        self.K = K
        self.T_n = np.zeros(K)
        self.t = 0
        self.is_done = False

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
        info = {}

        # action = None
        means = self.beliefs.sample()
        a = np.argmax(means)

        b = np.random.binomial(1, self.beta, 1)[0]

        d_n_delta = self.d_n_delta(1 - self.confidence, self.K)

        W = self.compute_W_n_i_j(self.beliefs.sigma, means)  # assume sigma=1
        # print("W: ", W)

        if b == 1:
            action = a
        else:
            a_idx = np.delete(np.array(range(self.K)), a)
            action = np.argmin(W[a, a_idx])

        if(not self.is_done):
            self.stopping_rule(d_n_delta, W, means)

        if(self.is_done):
            action = a

        self.t += 1
        self.T_n[action] += 1

        info['T_n'] = self.T_n
        info['t'] = self.t
        info['Posterior'] = None
        info['Model'] = None

        return action, self.is_done, info

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

        self.beliefs.T_n = self.T_n
        # for x_i, a_i, r_i in zip(x, action, reward):
        #    self.beliefs[a_i].update(x_i, r_i)
        self.beliefs.update(x, action, reward)

    def get_recommendation(self, x):
        """
        Sample rewards from the beliefs, return the max

         Returns:
            - recommendation (int): Index of the best arm according to the beliefs
        """
        # means = belief.sample()
        # recommendation = np.argmax(means)
        # print("recommendation: ", recommendation)
        #
        # return recommendation
        pass

    def stopping_rule(self, d_n_delta, W, means):
        i_val, i = np.max(means), np.argmax(means)
        means_j = np.delete(means, i)
        j = np.argmin(means_j)

        self.is_done = W[i, j] > d_n_delta

    def compute_W_n_i_j(self, sigma, mu_n):
        W = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(self.K):
                if(mu_n[j] < mu_n[i]):
                    W[i, j] = ((mu_n[i] - mu_n[j])**2) / \
                        (2 * (sigma**2)
                         * ((1 / self.T_n[i]) + (1 / self.T_n[j])))
        return W

    def CgG(self, x):
        return x + np.log(x)

    def d_n_delta(self, delta, K):
        return 4 * np.log(4 + np.log(self.t)) + 2 * self.CgG((np.log((K - 1) / (delta + PRECISION))) / 2)

    def c_n_delta(d_n_delta):
        return 1 - (1 / np.sqrt(2 * np.pi)) * np.exp(- ((np.sqrt(d_n_delta)) + (1 / np.sqrt(2))) ** 2)
