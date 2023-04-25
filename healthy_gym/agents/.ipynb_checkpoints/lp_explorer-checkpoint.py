from gurobipy import *
import numpy as np
from copy import deepcopy

import scipy.stats as stats


class LPExplorer:
    def __init__(self, models, belief, K=8, latent_dim=6, delta=0.05):
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
        self.means = None
        self.w_star = None
        self.t = 0
        self.latent_dim = latent_dim

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
        info = {}
        info['recommended_a_tau'] = None

        zz = kwargs.get('z', None)

        self.z_sample = self.belief.sample()
        info['Posterior'] = self.belief.prior

        # if zz is None:
        #     #posterior = deepcopy(self.belief.prior)
        #     #self.z_sample = np.argmax(posterior)
        #     self.z_sample = self.belief.sample()
        # else:
        #     self.z_sample = zz
        #     self.belief.prior = np.zeros(self.latent_dim) / self.latent_dim
        #     self.belief.prior[self.z_sample] = 1

        info['Model'] = self.z_sample

        j = np.argmin(self.N)

        if(self.N[j] <= np.sqrt(self.t) - self.K / 2):
            action = j
            self.N[action] += 1
        else:
            action = np.argmin(self.N - self.t * self.w_star[self.z_sample])
            self.N[action] += 1

        info['N'] = self.N

        #is_done = posterior[recommendation] > self.confidence

        # While not done
        if(not self.is_done):
            # Check stopping rule with the argmax in Z action
            self.stopping_rule(
                self.belief.prior, self.delta, np.argmax(self.models[self.z_sample].predict(x)))
            info['recommended_a_tau'] = action
        elif(self.is_done):
            action = np.argmax(self.models[self.z_sample].predict(x))

            # action = np.argmax(
            #    self.means[self.z_sample])
            info['recommended_a_tau'] = action
            # return action, self.is_done, info

        return action, self.is_done, info

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

    def Gaussian_KL(self, z_dim, means, sigma=None):
        if sigma is None:
            sigma = np.ones(z_dim)
        kl = [[]] * z_dim
        for z in range(z_dim):  # z to compare to;
            ll = []
            for zi in [x for x in list(range(z_dim)) if x != z]:  # other zs
                if ((np.argmax(means[z]) != np.argmax(means[zi]))):
                    mu_diff = means[z] - means[zi]
                    kl_term = np.log(
                        sigma[z] / sigma[zi]) + ((sigma[z] ** 2) + (mu_diff ** 2) / (2 * sigma[zi] ** 2)) - (1 / 2.0)
                    ll.append(kl_term)

            kl[z] = (ll)
        return kl

    def solve_lp(self, x, sigma=None):
        w_star = []
        I = np.ones(self.K)
        log_term = np.log(1 / (2.4 * self.delta))

        means = []
        for i in range(self.latent_dim):
            means.append(self.models[i].predict(x))
        self.means = np.array(np.squeeze(np.array(means)))

        KL = self.Gaussian_KL(self.latent_dim, self.means, sigma)

        for z in range(self.latent_dim):
            m = Model()
            m.Params.LogToConsole = 0  # Disable prints
            W = m.addMVar(self.K, lb=0, vtype=GRB.INTEGER, name="W")
            m.setObjective(I.T @ W)
            m.addConstr((np.array(KL[z]) @ W) - log_term >= 0)

            m.optimize()
            # for v in m.getVars():
            #    print(v.varName, v.x)
            w_star.append(W.x)

        return np.array(w_star), self.means

    def stopping_rule(self, posterior, delta, a_t):
        for z in range(self.latent_dim):
            a_star = np.argmax(self.means[z])
            if(a_star == a_t):
                self.Z_tau += posterior[z]
        self.is_done = (self.Z_tau > (1 - delta))
