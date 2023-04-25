import numpy as np
from .base_agent import BaseAgent


def xpluslogx(x):
    return x + np.log(x)


class ThompsonSampling(BaseAgent):
    def __init__(self, beliefs, explore=False, confidence=None, theta=0.5):
        """
        Standard Thompson Sampling (TS) for the Multi-armed Bandit assuming no structure between the actions

        Args:
            beliefs: A set of beliefs, one for each action.
            explore: Whether to do pure exploration or regret minimization.
            beta: Controls the re-sampling step of pure-exploration Thompson sampling. See https://arxiv.org/abs/1602.08448 for details.
            confidence: At which confidence level the algorithm should terminate in the pure-exploration setting.
                If None, the algorithm will never terminate.
        """
        super(ThompsonSampling, self).__init__()
        self.beliefs = beliefs
        self.explore = explore
        self.confidence = confidence
        self.theta = theta
        self.means = None

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
        if self.explore:
            b = np.random.binomial(1, self.theta, 1)[0]
            recommendation, confidence_level, is_done = self.get_recommendation(
                x)
            # if b==1:
            print("confidence_level, is_done: ", confidence_level, is_done)

            if is_done:
                info = {'samples': [belief.means(x) for belief in self.beliefs],
                        'statistics': [belief.statistics(x) for belief in self.beliefs],
                        'confidence_level': confidence_level}
                return recommendation, is_done, info
        else:
            confidence_level = None

        # Perform TS
        samples = np.array([belief.sample(x) for belief in self.beliefs])
        action = np.argmax(samples)

        info = {'samples': samples,
                'confidence_level': confidence_level}

        return action, False, info

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
            self.beliefs[a_i].update(x_i, r_i)

    def get_recommendation(self, x):
        """
        Check whether the beliefs are separated enough so that the algorithm can recommend an action (or model) with
         confidence at least self.confidence.

         Returns:
            - recommendation (int): Index of the best arm according to the beliefs
            - confidence_level (float): The probability of the recommendation being correct
            - is_done (boolean): True of confidence_level => self.confidence
        """
        means = [belief.sample(x) for belief in self.beliefs]
        print("means: ", means)

        recommendation = np.argmax(means)
        print("recommendation: ", recommendation)

        return recommendation, None, False


class ModelThompsonSampling(BaseAgent):

    def __init__(self, models, belief, explore=False, confidence=None, rule='simple'):
        """
        Standard Thompson Sampling (TS) for the Multi-armed Bandit assuming no structure between the actions

        Args:
            models: Set of possible models
            belief (ModelBelief): Belief over the models
            confidence: At which confidence level the algorithm should terminate in the pure-exploration setting.
                If None, the algorithm will never terminate.
            rule (string): Which decision rule to use
        """

        super(ModelThompsonSampling, self).__init__()
        #self.k = len(models)
        self.models = models
        self.explore = explore
        self.belief = belief
        self.rule = rule
        self.confidence = confidence
        self.t = 0

    def reset(self):
        self.t = 0

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
        if self.explore:
            recommendation, posterior, is_done = self.get_recommendation()

            info = {'Posterior': posterior, 'Model': recommendation}
            if is_done:
                return recommendation, is_done, info
        else:
            is_done = False
            info = {}

        model = self.belief.sample()
        y = self.models[model].predict(x)
        action = np.argmax(y)

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
            self.t += 1
        else:
            self.t += len(x)

        for x_i, a_i, r_i in zip(x, action, reward):
            self.belief.update(x_i, a_i, r_i)

    def get_recommendation(self):
        """
        Check whether the beliefs are separated enough so that the algorithm can recommend an action (or model) with
         confidence at least self.confidence.

         Returns:
            - recommendation (int): Index of the best arm according to the beliefs
            - is_done (boolean): True of confidence_level => self.confidence
        """
        if self.rule == 'simple':
            i, p = self.belief.get_map()
            if p > 1 - self.confidence:
                recommendation = i
                is_done = True
            else:
                recommendation = i
                is_done = False

        else:
            if self.t == 0:
                d = 0
            else:
                d = 4 * np.log(4 + np.log(self.t)) + 2 * \
                    xpluslogx(np.log((self.k - 1) / self.confidence) / 2)

            c = 1 - 1 / np.sqrt(1 * np.pi) * \
                np.exp(-(np.sqrt(d) + 1 / np.sqrt(2))**2)
            i, p = self.belief.get_map()
            if p > c:
                recommendation = i
                is_done = True
            else:
                recommendation = i
                is_done = False

        return recommendation, p, is_done


class ModelThompsonSampling2(BaseAgent):

    def __init__(self, model, belief, explore=False, confidence=None, rule='simple'):
        """
        Standard Thompson Sampling (TS) for the Multi-armed Bandit assuming no structure between the actions

        Args:
            models: Set of possible models
            belief (ModelBelief): Belief over the models
            confidence: At which confidence level the algorithm should terminate in the pure-exploration setting.
                If None, the algorithm will never terminate.
            rule (string): Which decision rule to use
        """

        super(ModelThompsonSampling2, self).__init__()
        #self.k = len(models)
        self.model = model
        self.explore = explore
        self.belief = belief
        self.rule = rule
        self.confidence = confidence
        self.t = 0

    def reset(self):
        self.t = 0

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
        if self.explore:
            recommendation, posterior, is_done = self.get_recommendation()

            info = {'Posterior': posterior, 'Model': recommendation}
            if is_done:
                return recommendation, is_done, info
        else:
            is_done = False
            info = {}

        z = kwargs.get('z', None)
        # print(z)
        if z is None:
            z = self.belief.sample()

        y = self.model.predict(x, z)
        action = np.argmax(y)
        info['z'] = z

        return action, is_done, info

    def update(self, x, z, action, reward, **kwargs):
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
            self.t += 1
        else:
            self.t += len(x)

        for x_i, a_i, r_i in zip(x, action, reward):
            self.belief.update(x_i, a_i, r_i)

    def get_recommendation(self):
        """
        Check whether the beliefs are separated enough so that the algorithm can recommend an action (or model) with
         confidence at least self.confidence.

         Returns:
            - recommendation (int): Index of the best arm according to the beliefs
            - is_done (boolean): True of confidence_level => self.confidence
        """
        if self.rule == 'simple':
            i, p = self.belief.get_map()
            if p > 1 - self.confidence:
                recommendation = i
                is_done = True
            else:
                recommendation = i
                is_done = False

        else:
            if self.t == 0:
                d = 0
            else:
                d = 4 * np.log(4 + np.log(self.t)) + 2 * \
                    xpluslogx(np.log((self.k - 1) / self.confidence) / 2)

            c = 1 - 1 / np.sqrt(1 * np.pi) * \
                np.exp(-(np.sqrt(d) + 1 / np.sqrt(2))**2)
            i, p = self.belief.get_map()
            if p > c:
                recommendation = i
                is_done = True
            else:
                recommendation = i
                is_done = False

        return recommendation, p, is_done
