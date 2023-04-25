import numpy as np
from .environment import Environment


class GaussianEnvironment(Environment):

    def __init__(self, means, std=1, seed=None):
        '''
        Ini environment

        Args:
            means: Npy array [Z, A] where Z is number of models and A number of arms
            std: Standard deviation of the noise
        '''
        self.means = means
        self.std = std
        self.Z = means.shape[0]
        self.random_state = np.random.RandomState(seed)
        self.true_z = self.random_state.choice(self.Z)

    def get_models(self):
        '''
        Return models as a Dummy class with a predict function

        Return:
            models: list of Model objects
        '''

        models = []
        for z in range(self.Z):
            models.append(Model(means=self.means[z, :]))

        return models

    def reset(self):
        '''
        Reset environment.
        '''
        self.true_z = self.random_state.choice(self.Z)

        return None

    def step(self, action):
        """
        Plays an action, returns a reward and updates or terminates the environment

        Args:
            action: Played action

        Returns:
            observation (object): The observation following the action (None if terminal state reached).
            reward (float): The reward of the submitted action
            done (boolean): True if the environment has reached a terminal state
            info (dict): Returns e.g., the reward distribution of all actions so that regret can be computed
        """

        simple_regret = np.max(
            self.means[self.true_z, :]) - self.means[self.true_z, action]

        reward = self.random_state.normal(
            self.means[self.true_z, action], self.std)

        info = {'regret': simple_regret,
                'reward': self.means[self.true_z, action]}

        return reward, reward, False, info

    def correct_model(self, model):
        return model == self.true_z

    def correct_arm(self, arm):
        return arm == np.argmax(self.means[self.true_z, :])

    def simple_regret(self, model):
        action = np.argmax(self.means[model, :])
        return np.max(self.means[self.true_z, :]) - self.means[self.true_z, action]


class Model:
    def __init__(self, means):
        '''
        Ini model

        Args:
            means: mean vector for the arms
        '''

        self.means = means

    def predict(self, x=None):
        '''
        Predict expected mean given context x. Context not used in this case.
        '''

        return self.means
