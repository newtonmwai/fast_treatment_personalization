from healthy_gym.environments.adcb import config
import numpy as np
import scipy.stats as stats
PRECISION = 1e-24


class BaseBelief:
    def __init__(self, **kwargs):
        """
        Belief over a set of parameters.

        """
        pass

    def sample(self, x):
        """
        Given a context x, draw a sample from the belief p(mu|x, H)

        Args:
            x: Current context

        Return:
            s: sample drawn from the belief distribution
        """

        raise Exception('Not implemented')

    def cdf(self, x, q):
        """
        Compute the cdf of the belief at point q given context x

        Args:
            x: Current context
            q: Value at which we evaluate the cdf

        Return:
            cdf (float): Value of the cdf at q
        """
        raise Exception('Not implemented')

    def pdf(self, x, q):
        """
        Compute the pdf of the belief at point q given context x

        Args:
            x: Current context
            q: Value at which we evaluate the cdf

        Return:
            pdf (float): Value of the pdf at q
        """
        raise Exception('Not implemented')

    def mean(self, x):
        """
        Returns the mean of the posterior given the context x.

        Args:
            x: Current context

        Return:
            mean (float): Mean of the belief distribution
        """
        raise Exception('Not implemented')

    def mode(self, x):
        """
        Returns the mode of the posterior given the context x.

        Args:
            x: Current context

        Return:
            mode (float): Mode of the belief distribution
        """
        raise Exception('Not implemented')

    def var(self, x):
        """
        Returns the var of the posterior given the context x.

        Args:
            x: Current context

        Return:
            var (float): Var of the belief distribution
        """
        raise Exception('Not implemented')

    def statistic(self, x):
        """
        Return the sufficient statistic (or approximate) of the belief distribution at x

        Args:
            x: Current context

        Return:
            sufficient (dict): Dictionary containing the sufficient statistic.
        """

        raise Exception('Not implemented')

    def update(self, x, observation):
        """
        Updates the belief given the context x and the observation

        Args:
            x: Current context
            observation: Observation from the environment
        """
        raise Exception('Not implemented')


class BetaBelief(BaseBelief):
    def __init__(self, alpha=1, beta=1, random_state=None):
        '''
        Defines a beta belief over [0, 1]. Default is a uniformed belief. The belief is oblivious to the context x.

        Args:
            s: Initial alpha parameter
            f: Initial beta parameter
            random_state: Seed for the sampling step
        '''

        super(BetaBelief, self).__init__()

        assert alpha > 0
        assert beta > 0

        self.alpha = alpha
        self.beta = beta
        self.random_state = np.random.RandomState(random_state)

    def sample(self, x):
        """
        Given a context x, draw a sample from the belief p(mu|x, H)

        Args:
            x: Current context

        Return:
            s: sample drawn from the belief distribution
        """

        return self.random_state.beta(self.alpha, self.beta, size=1)

    def cdf(self, x, q):
        """
        Compute the cdf of the belief at point q given context x

        Args:
            x: Current context
            q: Value at which we evaluate the cdf

        Return:
            pdf (float): Value of the cdf at q
        """
        return stats.beta.cdf(q, self.alpha, self.beta)

    def pdf(self, x, q):
        """
        Compute the cdf of the belief at point q given context x

        Args:
            x: Current context
            q: Value at which we evaluate the cdf

        Return:
            pdf (float): Value of the pdf at q
        """
        return stats.beta.pdf(q, self.alpha, self.beta)

    def var(self, x):
        """
        Returns the var of the posterior given the context x.

        Args:
            x: Current context

        Return:
            var (float): Var of the belief distribution
        """
        return stats.beta.var(self.alpha, self.beta)

    def mean(self, x):
        """
        Returns the mean of the posterior given the context x.

        Args:
            x: Current context

        Return:
            mean (float): Mean of the belief distribution
        """
        return stats.beta.mean(self.alpha, self.beta)

    def mode(self, x):
        """
        Returns the mode of the posterior given the context x.

        Args:
            x: Current context

        Return:
            mode (float): Mode of the belief distribution
        """
        assert self.alpha > 1
        assert self.beta > 1

        return (self.alpha - 1) / (self.alpha + self.beta - 2)

    def statistic(self, x):
        """
        Return the sufficient statistic (or approximate) of the belief distribution at x

        Args:
            x: Current context

        Return:
            sufficient (dict): Dictionary containing the sufficient statistic.
        """

        return {'alpha': self.alpha, 'beta': self.beta}

    def update(self, x, observation):
        """
        Updates the belief given the context x and the observation

        Args:
            x: Current context
            observation: Observation from the environment
        """

        self.alpha += observation
        self.beta += 1 - observation


class ModelBelief:
    def __init__(self, **kwargs):
        """
        Belief over a set of models.

        """
        pass

    def sample(self):
        """
        Given a context x, draw a sample from the belief p(mu|x, H)

        Args:
            x: Current context

        Return:
            s: sample drawn from the belief distribution
        """

        raise Exception('Not implemented')

    def update(self, x, action, observation):
        """
        Updates the belief given the context x and the observation

        Args:
            x: Current context
            observation: Observation from the environment
        """
        raise Exception('Not implemented')

    def get_map(self):
        """
        Returns the current most plausible model along with the probability

        """
        i = np.argmax(self.prior)
        return i, self.prior[i]

    def kl(self, param_1, param_2):
        '''
        Computes the KL-divergence between two distributions

        Args:
            param_1: Parameters defining the first distribution
            param_2: Parameters defining the second distribution

        Returns:
            kl_div: KL(P1 || P2)
        '''

        raise Exception('Not implemented')

    def kl_matrix(self, x):
        '''
        Returns the kl divergence between the distributions for arms in different models.

        Args:
            x: Current context

        Return:
            kl_mat: numpy matrix of size n_models x n_models  x arms. Index (i, j, k) corresponds to KL(k_i, k_j)
        '''

        kl_mat = []

        if self.models is None:
            Exception('Set of models does not exist!')

        means = np.array([model.predict(x)[0].flatten()
                         for model in self.models])
        n_arms = means.shape[1]
        for z, _ in enumerate(self.models):  # "True" model
            kl_z = []
            for z_hat, _ in enumerate(self.models):
                kl_hat = []  # Compare true model to each auxillary model
                for arm in range(n_arms):  # Compare kl-div for each arm
                    kl_hat.append(self.kl(means[z, arm], means[z_hat, arm]))
                kl_z.append(kl_hat)
            kl_mat.append(kl_z)

        return np.array(kl_mat)


class LatentModelBelief(ModelBelief):
    def __init__(self, models, prior=None, random_state=0, sigma=1, latent_dim=6):
        """
        Assumes that the reward from each arm is sampled according to N(Y, sigma)
        Latent state is sampled from z~Bern(theta) - we aim to update theta posterior


        Args:
            models: list of models that maps context x to expected behavior of the environment.
                Models should follow the scikit standards and have a model.predict(x) function.
            prior: Prior belief over the set of models. If None a uniform prior is used
            var: The variance of the reward distribution

        """
        self.sigma = sigma
        self.latent_dim = latent_dim

        super().__init__

        if prior is None:
            self.prior = np.ones(self.latent_dim) / self.latent_dim
        else:
            self.prior = prior

        self.t2 = [1]
        # self.posterior = self.prior
        self.models = models
        self.random_state = np.random.RandomState(random_state)
        self.p1 = self.prior
        # self.pz2 = 1 - self.posterior

    def sample(self):
        """
        Sample a Z from the posterior

        Return:
            z: 0 or 1
        """

        # z = self.random_state.binomial(size=1, n=1, p=self.posterior)
        return self.random_state.choice(self.latent_dim, p=self.prior, replace=True)

    def update(self, x, action, observation):
        """
        Updates the belief given the context x, z, and the observation

        Args:
            x: Current context
            action: action taken
            observation: Observation from the environment
        """
        p_ym = np.zeros(self.latent_dim)

        for z in range(self.latent_dim):
            #y = self.models.predict(x)

            # print('update: ', y)
            likelihood = stats.norm.pdf(
                observation, loc=self.models[z][action], scale=self.sigma)  # y[action] self.model[z][action]
            p_ym[z] = likelihood * self.prior[z] + PRECISION

        self.prior = p_ym / np.sum(p_ym)


# class GaussianBelief(BaseBelief):
#     def __init__(self, prior=None, random_state=0, sigma=1, K):
#         """
#         Assumes that the reward from each arm is sampled according to N(Y, sigma)
#
#
#         Args:
#             prior: Prior belief over the set of models. If None a uniform prior is used
#             var: The variance of the reward distribution
#
#         """
#         self.sigma = sigma
#
#         super().__init__
#
#         if prior is None:
#             self.prior = np.ones(self.latent_dim) / self.latent_dim
#         else:
#             self.prior = prior
#
#         self.t2 = [1]
#         # self.posterior = self.prior
#         self.models = models
#         self.random_state = np.random.RandomState(random_state)
#         self.p1 = self.prior
#         # self.pz2 = 1 - self.posterior
#
#     def sample(self):
#         """
#         Sample a Z from the posterior
#
#         Return:
#             z: 0 or 1
#         """
#
#         # z = self.random_state.binomial(size=1, n=1, p=self.posterior)
#         return self.random_state.choice(self.latent_dim, p=self.prior, replace=True)
#
#     def update(self, x, action, observation):
#         """
#         Updates the belief given the context x, z, and the observation
#
#         Args:
#             x: Current context
#             action: action taken
#             observation: Observation from the environment
#         """
#         p_ym = np.zeros(self.latent_dim)
#
#         for z in range(self.latent_dim):
#             #y = self.models.predict(x)
#
#             # print('update: ', y)
#             likelihood = stats.norm.pdf(
#                 observation, loc=self.models[z][action], scale=self.sigma)  # y[action] self.model[z][action]
#             p_ym[z] = likelihood * self.prior[z] + PRECISION
#
#         self.prior = p_ym / np.sum(p_ym)


class BernoulliModelBelief(ModelBelief):
    def __init__(self, models, prior=None, random_state=None):
        """
        Assumes that the reward from each arm is sampled according to Bern(model(x))

        Args:
            models: list of models that maps context x to expected behavior of the environment.
                Models should follow the scikit standards and have a model.predict(x) function.
            prior: Prior belief over the set of models. If None a uniform prior is used
            var: The variance of the reward distribution
        """
        super(BernoulliModelBelief, self).__init__()

        if prior is None:
            self.prior = np.ones(len(models)) / len(models)
        else:
            self.prior = prior

        self.models = models
        self.random_state = np.random.RandomState(random_state)

        if len(models) != len(self.prior):
            raise Exception('Length of models and prior do not match')

    def sample(self):
        """
        Sample a model from the posterior

        Return:
            index: Index of the sampled model
        """

        return self.random_state.choice(len(self.models), p=self.prior)

    def update(self, x, action, observation):
        """
        Updates the belief given the context x and the observation

        Args:
            x: Current context
            action: action taken
            observation: Observation from the environment
        """

        p_ym = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            y = model.predict(x)
            likelihood = stats.bernoulli.pmf(observation, y[action])
            p_ym[i] = likelihood * self.prior[i]

        self.prior = p_ym / np.sum(p_ym)


class GaussianModelBelief(ModelBelief):
    def __init__(self, models, prior=None, std=1, seed=0, contextualReward=False):
        """
        Assumes that the reward from each arm is sampled according to N(model(x), var) for some model in models.
        The variance is assumed to be known.

        Args:
            models: list of models that maps context x to expected behavior of the environment.
                Models should follow the scikit standards and have a model.predict(x) function.
            prior: Prior belief over the set of models.
            rmse: The std of the reward distribution
            seed: Seed for npy random state

        """
        super(GaussianModelBelief, self).__init__()

        if prior is None:
            self.prior = np.ones(len(models)) / len(models)
        else:
            self.prior = prior

        self.models = models
        self.std = std
        self.random_state = np.random.RandomState(seed)

        self.contextualReward = contextualReward

        self.y = None
        self.av45 = None
        self.fdg = None
        self.tau = None
        self.ptau = None

    def sample(self):
        """
        Sample a model from the posterior

        Return:
            index: Index of the sampled model
        """
        return self.random_state.choice(len(self.models), p=self.prior)

    def update(self, x, action, observation):
        """
        Updates the belief given the context x and the observation

        Args:
            x: Current context
            action: action taken
            observation: Observation from the environment
        """
        p_ym = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            # print(x.columns)
            #y, av45, fdg = model.predict(x)
            likelihood = stats.norm.pdf(
                observation, loc=self.y[i][action], scale=self.std)

            likelihood_av45 = stats.norm.pdf(
                self.av45[i][0], loc=self.av45[i][1], scale=config.Autoreg_AV45_NOISE)
            likelihood_fdg = stats.norm.pdf(
                self.fdg[i][0], loc=self.fdg[i][1], scale=config.Autoreg_FDG_NOISE)
            likelihood_tau = stats.norm.pdf(
                self.tau[i][0], loc=self.tau[i][1], scale=config.Autoreg_TAU_NOISE)
            likelihood_ptau = stats.norm.pdf(
                self.ptau[i][0], loc=self.ptau[i][1], scale=config.Autoreg_PTAU_NOISE)

            # if(self.contextualReward):
            #

            p_ym[i] = likelihood * self.prior[i] * likelihood_av45 * \
                likelihood_fdg * likelihood_tau * likelihood_ptau + PRECISION
            # else:
            #p_ym[i] = likelihood * self.prior[i] + PRECISION

        self.prior = p_ym / np.sum(p_ym)

    def kl(self, param1, param2):
        '''
        KL divergence between two Gaussians with same variance
            Args:
            param_1: Mean defining the first distribution
            param_2: Mean defining the second distribution

        Returns:
            kl_div: KL(P1 || P2)
        '''

        return ((param1 - param2) ** 2) / (2 * (self.std ** 2))

    def sample_reward(self, x, a, n_draws=1):
        '''
        Sample n_draws from arm a  with context x,
        '''
        samples = []
        for i in range(n_draws):
            z = self.random_state.choice(len(self.prior), p=self.prior)
            reward = self.random_state.normal(
                self.y[z][a], scale=self.std)  # self.models[z].predict(x)[a]
            samples.append(reward)
        return samples

    def pseudo_update(self, x, action, observation):
        """
        Return the updated belief without saving it to the object.

        Args:
            x: Current context
            action: action taken
            observation: Observation from the environment
        """
        """p_ym = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            y = model.predict(x)
            likelihood = stats.norm.pdf(
                observation, loc=y[action], scale=self.std)
            p_ym[i] = likelihood * self.prior[i] + PRECISION
        prior = p_ym / np.sum(p_ym)
        return prior"""

        p_ym = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            # print(x.columns)
            #y, av45, fdg = model.predict(x)
            likelihood = stats.norm.pdf(
                observation, loc=self.y[i][action], scale=self.std)
            likelihood_av45 = stats.norm.pdf(
                self.av45[i][0], loc=self.av45[i][1], scale=config.Autoreg_AV45_NOISE)
            likelihood_fdg = stats.norm.pdf(
                self.fdg[i][0], loc=self.fdg[i][1], scale=config.Autoreg_FDG_NOISE)
            likelihood_tau = stats.norm.pdf(
                self.tau[i][0], loc=self.tau[i][1], scale=config.Autoreg_TAU_NOISE)
            likelihood_ptau = stats.norm.pdf(
                self.ptau[i][0], loc=self.ptau[i][1], scale=config.Autoreg_PTAU_NOISE)

            # if(self.contextualReward):
            #
            p_ym[i] = likelihood * self.prior[i] * likelihood_av45 * \
                likelihood_fdg * likelihood_tau * likelihood_ptau + PRECISION
            # else:
            #p_ym[i] = likelihood * self.prior[i] + PRECISION

        prior = p_ym / np.sum(p_ym)
        #print("pseudo prior: ", prior)
        return prior


class LinearGaussianBelief(BaseBelief):
    '''
    Belief used for standard contextual Linear TS with one theta per arm. Note that you have to use one LinearGaussianBelief per arm.

    To Do: add possibility to use informative prior
    '''

    def __init__(self, d, v=1, means=None, seed=None):
        '''
        Ini model

        Args:
            d: dimension of space
            k: number of arms
            v: Controls the scaling of the variance (See Agrawal Goyal 2013)
            means: Mean vector. If none ini to 0
            seed: Seed to npy Random state
        '''
        super(LinearGaussianBelief, self).__init__()
        self.d = d
        self.v = v
        self.random_state = np.random.RandomState(seed)

        self.B = np.eye(d)

        if means is None:
            self.mu = np.zeros([d, 1])
            self.f = np.zeros([d, 1])
        else:
            dim, t = means.shape
            if t != 1:
                Exception('Wrong dimensionality of mean. Should be (d, 1)')
            self.mu = means
            self.f = means

    def sample(self, x):
        '''
        Given context x sample mean rewards for each arm.

        B will be inverted using Monroe-Penrose to ensure existence

        Args:
            x: Current context

        Returns:
            rewards: expected reward
        '''

        B_inv = np.linalg.pinv(self.B)
        mu_hat = self.random_state.multivariate_normal(
            self.mu.reshape(self.d, ), (self.v ** 2) * B_inv)
        reward = np.dot(x, mu_hat)
        return reward

    def update(self, x, observation):
        '''
        Update Gaussian belief

        B will be inverted using Monroe-Penrose to ensure existence

        Args:
            x: current context
            observation: observation given action and x
        '''

        if len(x.shape) == 1:
            x = x.reshape(self.d, 1)
        xxT = x @ x.T
        if xxT.shape[0] != self.d or xxT.shape[1] != self.d:
            Exception('Wrong dimension on xx^T')

        self.B = self.B + xxT
        self.f = self.f + x * observation
        self.mu = np.linalg.pinv(self.B) @ self.f


class GaussianBelief(BaseBelief):
    '''
    Belief used for TTTS with one theta per arm. Note that you have to use one GaussianBelief per arm.

    To Do: add possibility to use informative prior
    '''

    def __init__(self, K=8, seed=None, sigma=9.5):
        '''
        Ini model

        Args:
            v: Controls the scaling of the variance (See Agrawal Goyal 2013)
            means: Mean vector. If none ini to 0
            seed: Seed to npy Random state
        '''
        super(GaussianBelief, self).__init__()
        self.random_state = np.random.RandomState(seed)

        self.K = K
        self.mean = np.zeros(self.K)
        self.var = np.ones(self.K)

        self.sigma = sigma

        self.T_n = np.zeros(K)
        self.Y_n = np.zeros(K)

    def sample(self):
        '''
        Args:
            : Current context

        Returns:
            rewards: expected reward
        '''
        return np.array([self.random_state.normal(self.mean[i], self.var[i]) for i in range(self.K)])

    def update(self, x, action, reward):
        '''
        Update Gaussian belief

        Args:
            x: current context
            observation: observation given action and x

        '''
        #print(self.Y_n[action], reward[0])
        self.Y_n[action] += reward[0]
        self.mean[action] = self.Y_n[action] / (self.T_n[action])
        self.var[action] = self.sigma / (self.T_n[action])
