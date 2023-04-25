import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from healthy_gym.agents.lp_explorer import *
from healthy_gym.agents.bayes_design import *
from healthy_gym.agents.beliefs import *
from healthy_gym.environments.adcb import *
from healthy_gym.agents.pure_explorer import *
from healthy_gym.agents.thompson_sampling import *
PRECISION = 1e-24


class RegretMinExperiment:
    '''
    Experiment class for regret minimization. Wraps the agent and environment. Provides all the important logs
    '''

    def __init__(self, agent, environment, n_steps=100):
        '''
        Init exp
        '''
        self.agent = agent
        self.environment = environment
        self.n_steps = n_steps

    def run(self):
        '''
        run exp
        '''
        log = {'t': [], 'reward': [], 'regret': [], 'action': []}
        x = self.environment.reset()
        for t in range(self.n_steps):
            action, _, _ = self.agent.act(x=x)
            x, reward, _, info = self.environment.step(action)
            self.agent.update(x, action, reward)

            # update log:
            log['t'].append(t)
            log['reward'].append(info['reward'])
            log['regret'].append(info['regret'])
            log['action'].append(action)

        log['cumulative_regret'] = np.cumsum(log['regret'])

        return log


class PureExploreExperiment:
    '''
    Experiment class for pure exploration. Wraps the agent and environment. Provides all the important logs
    '''

    def __init__(self, agent, environment, max_steps=10000):
        '''
        Init exp
        '''
        self.agent = agent
        self.environment = environment
        self.max_steps = max_steps

    def run(self):
        '''
        run exp
        '''
        t = 0
        is_done = False
        x = self.environment.reset()
        log = {'t': [], 'reward': [], 'expected reward': [],
               'regret': [], 'action': []}
        while t <= self.max_steps and not is_done:
            action, is_done, agent_info = self.agent.act(x=x)
            x, reward, _, info = self.environment.step(action)
            if not is_done:
                t += 1
                log['t'].append(t)
                log['expected reward'].append(info['reward'])
                log['reward'].append(reward)
                log['regret'].append(info['regret'])
                log['action'].append(action)
                self.agent.update(x, action, reward)

        recommendation = agent_info['Model']

        correct_model = self.environment.correct_model(recommendation)
        correct_arm = self.environment.correct_arm(action)

        log['Recommendation'] = recommendation
        log['Correct Model'] = correct_model
        log['Correct Arm'] = correct_arm

        log['Stop time'] = t

        return log


class PureExploreExperimentADCB:
    '''
    Experiment class for pure exploration. Wraps the agent and environment. Provides all the important logs
    '''

    def __init__(self, agent, environment, max_steps=100, exptype=None):
        '''
        Init exp
        '''
        self.agent = agent
        self.environment = environment
        self.max_steps = max_steps
        self.means = self.agent.means
        self.exptype = exptype

    def run(self):
        '''
        run exp
        '''
        t = 0
        is_done = False
        x = self.environment.buffer_

        z = int(x['Z'].values)
        xid = int(x['RID'].values)
        self.agent.xid = xid

        # if(self.exptype == "linTS"):
        #     x = np.array(dm.check_categorical(x, list(x.columns), '_'))
        #     x = x.reshape(30,)  # dimension=30

        a_star = np.argwhere(self.means[z] == np.amax(self.means[z]))

        log = {'t': [], 'reward': [], 'expected reward': [],
               'regret': [], 'action': [], 'posterior': np.ones(self.max_steps)}
        while t < self.max_steps and not is_done:
            action, is_done, agent_info = self.agent.act(x=x)
            _, reward, _, info = self.environment.step(action)
            # if not is_done:
            # print(t)
            log['t'].append(t)
            if agent_info['Posterior'] is not None:
                log['posterior'][t] = max(agent_info['Posterior'])
            log['expected reward'].append(info['reward'])
            log['reward'].append(reward)
            log['regret'].append(info['regret'])
            log['action'].append(action)
            self.agent.update(x, action, reward)
            t += 1

            # print("\nPosterior: ", self.agent.belief.prior)

        if agent_info['Model'] is not None:
            recommendation = int(agent_info['Model'])
        else:
            recommendation = -1

        correct_model = (recommendation == z)

        a_star = np.argmax(self.means[z])

        recommended_a = int(action)
        correct_a = int(recommended_a == a_star)

        if agent_info['Posterior'] is not None:
            log['posterior'] = np.array(log['posterior'])  # [-1]

        log['Recommendation'] = recommendation
        log['Correct Action'] = correct_a

        log['Recommended Action'] = recommended_a
        log['Correct Model'] = correct_model
        log['True State'] = z
        log['Stop time'] = t

        return log


def run_PE_experiments_baseline(confidence, max_steps, Samples, contextualR, prior, std):
    w_star_log = []
    kl_log = []
    ttts_log = []
    greedy_log = []
    lpexp_log = []

    # print("\nConfidence: ", confidence, " ... ")
    env = ADCBEnvironment(
        gamma=2,
        epsilon=0.1,
        policy='DX_Based',
        regenerate=False,
        horizon=6,
        n_buffer=10000,
        sequential=False,
        reward_sigma=std,
        contextualReward=contextualR,
        z_dim=6)

    for i in range(Samples):
        env.reset()
        x = env.buffer_
         models = env.get_models(env, contextualR=contextualR)

        # LP Explorer
         belief_lp = GaussianModelBelief(
            models=models, prior=prior, std=std, contextualReward=contextualR)
        # belief = LatentModelBelief(models=models)
        lp_explorer = DTrackingLPExplorer(
            models=models, belief=belief_lp, delta=1 - confidence)
        # solve LP
        w_star, pred_means, av45, fdg, tau, ptau = lp_explorer.solve_lp(
            x, sigma=std)

        belief_lp.y = pred_means
        belief_lp.av45 = av45
        belief_lp.fdg = fdg
        belief_lp.tau = tau
        belief_lp.ptau = ptau

        # belief.models = means
        exp3 = PureExploreExperimentADCB(
            agent=lp_explorer, environment=env, max_steps=max_steps)
        log3 = exp3.run()
         lpexp_log.append(log3)

        # Vanilla TTTS
        beliefs_vanillaTTTS = GaussianBelief(sigma=std)
        vanillaTTTS = VanillaTTTSExplorer(
            beliefs=beliefs_vanillaTTTS, confidence=confidence)
        vanillaTTTS.means = pred_means  # No use
        exp4 = PureExploreExperimentADCB(
            agent=vanillaTTTS, environment=env, max_steps=max_steps)
        log4 = exp4.run()
        greedy_log.append(log4)

        lpexp_results = pd.DataFrame(lpexp_log)
        greedy_results = pd.DataFrame(greedy_log)

     return lpexp_results, greedy_results


def run_PE_experiments(confidence, max_steps, Samples, contextualR, prior, std):
    w_star_log = []
    kl_log = []
    ttts_log = []
    greedy_log = []
    lpexp_log = []
    z_log = []

    deltta = 1 - confidence
    delttac = confidence

    kl_delta = np.log(1.0 / (2.4 * deltta + PRECISION))

    # print("\nConfidence: ", confidence, " ... ")
    env = ADCBEnvironment(
        gamma=2,
        epsilon=0.1,
        policy='DX_Based',
        regenerate=False,
        horizon=6,
        n_buffer=10000,
        sequential=False,
        reward_sigma=std,
        contextualReward=contextualR,
        z_dim=6)

    for i in range(Samples):
        env.reset()
        x = env.buffer_
        # print("Patient: ", i)
        models = env.get_models(env, contextualR=contextualR)

        # LP Explorer
        # print("\nLP Explorer...")
        belief_lp = GaussianModelBelief(
            models=models, prior=prior, std=std, contextualReward=contextualR)
        # belief = LatentModelBelief(models=models)
        lp_explorer = DTrackingLPExplorer(
            models=models, belief=belief_lp, delta=1 - confidence)
        # solve LP
        w_star, pred_means, av45, fdg, tau, ptau = lp_explorer.solve_lp(
            x, sigma=std)
        # print("\nconfidence, w_star", confidence, w_star)

        belief_lp.y = pred_means
        belief_lp.av45 = av45
        belief_lp.fdg = fdg
        belief_lp.tau = tau
        belief_lp.ptau = ptau
        # print(w_star)
        z_log.append(int(x['Z'].values))


        lb = kl_delta * np.sum(w_star[int(x['Z'].values)])

        # print("LB: ", lb)
        w_star_log.append(lb)

        # belief.models = means
        exp3 = PureExploreExperimentADCB(
            agent=lp_explorer, environment=env, max_steps=max_steps)
        log3 = exp3.run()
        # log2['Simple Regret'] = env.simple_regret(log2['Recommendation'])
        lpexp_log.append(log3)

        # Divergence Explorer
        belief_div = GaussianModelBelief(
            models=models, prior=prior, std=std, contextualReward=contextualR)
        belief_div.y = pred_means
        belief_div.av45 = av45
        belief_div.fdg = fdg
        belief_div.tau = tau
        belief_div.ptau = ptau
        kl_agent = DivergenceExplorer(
            models=models, belief=belief_div, confidence=confidence)
        kl_agent.means = pred_means
        exp1 = PureExploreExperimentADCB(
            agent=kl_agent, environment=env, max_steps=max_steps)
        log1 = exp1.run()
        # log1['Simple Regret'] = env.simple_regret(log1['Recommendation'])
        kl_log.append(log1)

        # Two-Top TS
        # print("\nTwo-Top TS Explorer...")
        belief_ttts = GaussianModelBelief(
            models=models, prior=prior, std=std, contextualReward=contextualR)
        belief_ttts.y = pred_means
        belief_ttts.av45 = av45
        belief_ttts.fdg = fdg
        belief_ttts.tau = tau
        belief_ttts.ptau = ptau
        ttts = TopTwoThompsonSampling(
            models=models, belief=belief_ttts, confidence=confidence)
        ttts.means = pred_means
        exp2 = PureExploreExperimentADCB(
            agent=ttts, environment=env, max_steps=max_steps)
        log2 = exp2.run()
        # log2['Simple Regret'] = env.simple_regret(log2['Recommendation'])
        ttts_log.append(log2)

        # Greedy Design
        # print("\nGreedy Explorer...")
        belief_greedy = GaussianModelBelief(
            models=models, prior=prior, std=std, contextualReward=contextualR)
        belief_greedy.y = pred_means
        belief_greedy.av45 = av45
        belief_greedy.fdg = fdg
        belief_greedy.tau = tau
        belief_greedy.ptau = ptau
        greedy = GreedyExplorer(
            models=models, belief=belief_greedy, delta=1 - confidence)
        greedy.means = pred_means
        exp4 = PureExploreExperimentADCB(
            agent=greedy, environment=env, max_steps=max_steps)
        log4 = exp4.run()
        greedy_log.append(log4)

        # Vanilla TTTS
        # beliefs_vanillaTTTS = GaussianBelief(sigma=std)
        # vanillaTTTS = VanillaTTTSExplorer(
        #    beliefs=beliefs_vanillaTTTS, confidence=confidence)
        # vanillaTTTS.means = pred_means  # No use
        # exp4 = PureExploreExperimentADCB(
        #    agent=vanillaTTTS, environment=env, max_steps=max_steps)
        # log4 = exp4.run()
        # greedy_log.append(log4)

        kl_results = pd.DataFrame(kl_log)
        ttts_results = pd.DataFrame(ttts_log)
        lpexp_results = pd.DataFrame(lpexp_log)
        greedy_results = pd.DataFrame(greedy_log)

        # lpexp_time2 = lpexp_results2['Stop time'].values
    return kl_results, ttts_results, lpexp_results, greedy_results, w_star_log


def collate_results_confidence__baseline(confidence, max_steps, Samples, contextualR, prior, std, w_stars,
                                         results_div, results_ttts, results_lp, results_greedy,
                                         results_div_correct, results_ttts_correct, results_lp_correct, results_greedy_correct,
                                         results_div_posterior, results_ttts_posterior, results_lp_posterior, results_greedy_posterior,
                                         results_div_posterior_err, results_ttts_posterior_err, results_lp_posterior_err, results_greedy_posterior_err,
                                         results_div_err, results_ttts_err, results_lp_err, results_greedy_err):

    lpexp_results, greedy_results = run_PE_experiments_baseline(
        confidence, max_steps, Samples, contextualR, prior, std)

    lpexp_time = lpexp_results['Stop time'].values
    greedy_time = greedy_results['Stop time'].values

    lp_correct = lpexp_results['Correct Action'].values
    greedy_correct = greedy_results['Correct Action'].values

    results_lp[confidence] = np.mean(lpexp_time)
    results_greedy[confidence] = np.mean(greedy_time)

    results_lp_correct[confidence] = np.mean(lp_correct)
    results_greedy_correct[confidence] = np.mean(greedy_correct)

    results_lp_err[confidence] = np.std(lpexp_time)
    results_greedy_err[confidence] = np.std(greedy_time)

    results_lp_posterior[confidence] = np.mean(
        lpexp_results['posterior'].values)
    results_greedy_posterior[confidence] = np.mean(
        greedy_results['posterior'].values)
    # print("results_div", results_div_posterior)

    results_lp_posterior_err[confidence] = np.std(
        lpexp_results['posterior'].values)
    results_greedy_posterior_err[confidence] = np.std(
        greedy_results['posterior'].values)


def collate_results_confidence(confidence, max_steps, Samples, contextualR, prior, std, w_stars,
                               results_div, results_ttts, results_lp, results_greedy,
                               results_div_correct, results_ttts_correct, results_lp_correct, results_greedy_correct,
                               results_div_posterior, results_ttts_posterior, results_lp_posterior, results_greedy_posterior,
                               results_div_posterior_err, results_ttts_posterior_err, results_lp_posterior_err, results_greedy_posterior_err,
                               results_div_err, results_ttts_err, results_lp_err, results_greedy_err):
    div_results, ttts_results, lpexp_results, greedy_results, w_star_log = run_PE_experiments(
        confidence, max_steps, Samples, contextualR, prior, std)

    div_time = div_results['Stop time'].values
    ttts_time = ttts_results['Stop time'].values
    lpexp_time = lpexp_results['Stop time'].values
    greedy_time = greedy_results['Stop time'].values

    div_correct = div_results['Correct Action'].values
    ttts_correct = ttts_results['Correct Action'].values
    lp_correct = lpexp_results['Correct Action'].values
    greedy_correct = greedy_results['Correct Action'].values

    results_div[confidence] = np.mean(div_time)
    results_ttts[confidence] = np.mean(ttts_time)
    results_lp[confidence] = np.mean(lpexp_time)
    results_greedy[confidence] = np.mean(greedy_time)

    results_div_correct[confidence] = np.mean(div_correct)
    results_ttts_correct[confidence] = np.mean(ttts_correct)
    results_lp_correct[confidence] = np.mean(lp_correct)
    results_greedy_correct[confidence] = np.mean(greedy_correct)

    results_div_err[confidence] = np.std(div_time)
    results_ttts_err[confidence] = np.std(ttts_time)
    results_lp_err[confidence] = np.std(lpexp_time)
    results_greedy_err[confidence] = np.std(greedy_time)

    results_div_posterior[confidence] = np.mean(
        div_results['posterior'].values)
    results_ttts_posterior[confidence] = np.mean(
        ttts_results['posterior'].values)
    results_lp_posterior[confidence] = np.mean(
        lpexp_results['posterior'].values)
    results_greedy_posterior[confidence] = np.mean(
        greedy_results['posterior'].values)
    # print("results_div", results_div_posterior)

    results_div_posterior_err[confidence] = np.std(
        div_results['posterior'].values)
    results_ttts_posterior_err[confidence] = np.std(
        ttts_results['posterior'].values)
    results_lp_posterior_err[confidence] = np.std(
        lpexp_results['posterior'].values)
    results_greedy_posterior_err[confidence] = np.std(
        greedy_results['posterior'].values)

    # print("w_star mean", np.mean(w_star_log))
    w_stars[confidence] = np.mean(w_star_log)


def collate_results_noise(confidence, max_steps, Samples, contextualR, prior, noise_std, w_stars,
                          results_div, results_ttts, results_lp, results_greedy,
                          results_div_correct, results_ttts_correct, results_lp_correct, results_greedy_correct,
                          results_div_posterior, results_ttts_posterior, results_lp_posterior, results_greedy_posterior,
                          results_div_posterior_err, results_ttts_posterior_err, results_lp_posterior_err, results_greedy_posterior_err,
                          results_div_err, results_ttts_err, results_lp_err, results_greedy_err):
    div_results, ttts_results, lpexp_results, greedy_results, w_star_log = run_PE_experiments(
        confidence, max_steps, Samples, contextualR, prior, noise_std)
    div_time = div_results['Stop time'].values
    ttts_time = ttts_results['Stop time'].values
    lpexp_time = lpexp_results['Stop time'].values
    greedy_time = greedy_results['Stop time'].values

    div_correct = div_results['Correct Action'].values
    ttts_correct = ttts_results['Correct Action'].values
    lp_correct = lpexp_results['Correct Action'].values
    greedy_correct = greedy_results['Correct Action'].values

    results_div[noise_std] = np.mean(div_time)
    results_ttts[noise_std] = np.mean(ttts_time)
    results_lp[noise_std] = np.mean(lpexp_time)
    results_greedy[noise_std] = np.mean(greedy_time)

    results_div_correct[noise_std] = np.mean(div_correct)
    results_ttts_correct[noise_std] = np.mean(ttts_correct)
    results_lp_correct[noise_std] = np.mean(lp_correct)
    results_greedy_correct[noise_std] = np.mean(greedy_correct)

    results_div_posterior[noise_std] = np.mean(
        div_results['posterior'].values)
    results_ttts_posterior[noise_std] = np.mean(
        ttts_results['posterior'].values)
    results_lp_posterior[noise_std] = np.mean(
        lpexp_results['posterior'].values)
    results_greedy_posterior[noise_std] = np.mean(
        greedy_results['posterior'].values)

    results_div_err[noise_std] = np.std(div_time)
    results_ttts_err[noise_std] = np.std(ttts_time)
    results_lp_err[noise_std] = np.std(lpexp_time)
    results_greedy_err[noise_std] = np.std(greedy_time)

    results_div_posterior_err[noise_std] = np.std(
        div_results['posterior'].values)
    results_ttts_posterior_err[noise_std] = np.std(
        ttts_results['posterior'].values)
    results_lp_posterior_err[noise_std] = np.std(
        lpexp_results['posterior'].values)
    results_greedy_posterior_err[noise_std] = np.std(
        greedy_results['posterior'].values)
    w_stars[noise_std] = np.mean(w_star_log)


def experiment_sigma(Samples, max_steps, confidence, noises, P_Z):
    experiments = [
        # {'Contextual': False, 'Prior': None},
        {'Contextual': True, 'Prior': None}
        # {'Contextual': False, 'Prior': P_Z},
        # {'Contextual': True, 'Prior': P_Z}
    ]

    for experiment in experiments:
        contextualR = experiment['Contextual']
        prior = experiment['Prior']
        # r_sigma = experiment['R_sigma'] #environent noise
        # std=experiment['std'] #Likelihood sigma

        filename = {}
        filename['Samples'] = Samples
        filename['Contextual'] = experiment['Contextual']
        filename['Prior'] = True if (
            experiment['Prior'] is not None) else False
        # filename['Rsigma'] = experiment['R_sigma']
        # filename['std'] = experiment['std']
        print(filename)

        w_stars = {}
        results_div = {}
        results_ttts = {}
        results_lp = {}
        results_greedy = {}

        results_div_correct = {}
        results_ttts_correct = {}
        results_lp_correct = {}
        results_greedy_correct = {}

        results_div_posterior = {}
        results_ttts_posterior = {}
        results_lp_posterior = {}
        results_greedy_posterior = {}

        results_div_err = {}
        results_ttts_err = {}
        results_lp_err = {}
        results_greedy_err = {}

        results_div_posterior_err, results_ttts_posterior_err, results_lp_posterior_err, results_greedy_posterior_err = {}, {}, {}, {}

        _ = list(map(lambda n_std: collate_results_noise(confidence, max_steps, Samples,
                                                         contextualR, prior, n_std, w_stars,
                                                         results_div, results_ttts, results_lp, results_greedy,
                                                         results_div_correct, results_ttts_correct, results_lp_correct, results_greedy_correct,
                                                         results_div_posterior, results_ttts_posterior, results_lp_posterior, results_greedy_posterior,
                                                         results_div_posterior_err, results_ttts_posterior_err, results_lp_posterior_err, results_greedy_posterior_err,
                                                         results_div_err, results_ttts_err, results_lp_err, results_greedy_err), noises))

        results_groups_correct = [[results_lp_correct[n_std], results_div_correct[n_std],
                                   results_ttts_correct[n_std], results_greedy_correct[n_std]] for n_std in noises]

        print('Correctness: ', results_groups_correct)
        results_groups = [[results_lp[n_std], results_div[n_std],
                           results_ttts[n_std], results_greedy[n_std]] for n_std in noises]
        print('Stop times: ', results_groups)

        results_groups_err = [[results_lp_err[n_std], results_div_err[n_std],
                               results_ttts_err[n_std], results_greedy_err[n_std]] for n_std in noises]

        lbs = [w_stars[n_std] for n_std in noises]

        ###
        plot_one_bar(noises, results_groups, r'$E[\tau]$', r'$\sigma^2$',
                     ['LLPT Explorer', 'Divergence Explorer',
                         'TTTS-Latent Explorer', 'Greedy Explorer'],
                     ylim=[0, 100], opacity=[0.1, 0.1, 1], filename=filename, lbs=lbs, results_groups_err=results_groups_err)

        plot_one_bar(noises, results_groups_correct, 'Correctness', r'$\sigma^2$',
                     ['LLPT Explorer', 'Divergence Explorer',
                         'TTTS-Latent Explorer', 'Greedy Explorer'],
                     ylim=[0, 1], opacity=[0.1, 0.1, 1], filename=filename)


def experiment_confidence_baseline(Samples, max_steps, confidences, sigma, P_Z):
    experiments = [
        # {'Contextual': False, 'Prior': None},
        {'Contextual': True, 'Prior': None}
        # {'Contextual': False, 'Prior': P_Z},
        # {'Contextual': True, 'Prior': P_Z},
    ]

    for experiment in experiments:
        contextualR = experiment['Contextual']
        prior = experiment['Prior']
        # r_sigma = experiment['R_sigma'] #environent noise
        # std=experiment['std'] #Likelihood sigma

        filename = {}
        filename['Samples'] = Samples
        filename['Contextual'] = experiment['Contextual']
        filename['Prior'] = True if (
            experiment['Prior'] is not None) else False
        # filename['Rsigma'] = experiment['R_sigma']
        # filename['std'] = experiment['std']
        print(filename)

        w_stars = {}
        results_div = {}
        results_ttts = {}
        results_lp = {}
        results_greedy = {}

        results_div_correct = {}
        results_ttts_correct = {}
        results_lp_correct = {}
        results_greedy_correct = {}

        results_div_posterior = {}
        results_ttts_posterior = {}
        results_lp_posterior = {}
        results_greedy_posterior = {}

        results_div_err, results_ttts_err, results_lp_err, results_greedy_err = {}, {}, {}, {}
        results_div_posterior_err, results_ttts_posterior_err, results_lp_posterior_err, results_greedy_posterior_err = {}, {}, {}, {}

        _ = list(map(lambda c: collate_results_confidence__baseline(c, max_steps, Samples,
                                                                    contextualR, prior, sigma, w_stars,
                                                                    results_div, results_ttts, results_lp, results_greedy,
                                                                    results_div_correct, results_ttts_correct, results_lp_correct, results_greedy_correct,
                                                                    results_div_posterior, results_ttts_posterior, results_lp_posterior, results_greedy_posterior,
                                                                    results_div_posterior_err, results_ttts_posterior_err, results_lp_posterior_err, results_greedy_posterior_err,
                                                                    results_div_err, results_ttts_err, results_lp_err, results_greedy_err), confidences))

        results_groups_correct = [
            [results_lp_correct[c], results_greedy_correct[c]] for c in confidences]
        print('Correctness: ', results_groups_correct)
        results_groups = [[results_lp[c], results_greedy[c]]
                          for c in confidences]
        print('Stop times: ', results_groups)

        # plot_one_bar(confidences, results_groups, r'$E[\tau]$', r'$1-\delta$',
        #              ['LLPT (With latent structure)',
        #               'TTTS (No latent structure)'],
        #              ylim=[0, 250], opacity=[0.1, 0.1, 1], filename=filename)
        #
        # plot_one_bar(confidences, results_groups_correct, 'Correctness', r'$1-\delta$',
        #              ['LLPT (With latent structure)',
        #               'TTTS (No latent structure)'],
        #              ylim=[0, 1], opacity=[0.1, 0.1, 1], filename=filename)

        return results_groups, results_groups_correct


def experiment_confidence(Samples, max_steps, confidences, sigma, P_Z):
    experiments = [
        # {'Contextual': False, 'Prior': None},
        {'Contextual': True, 'Prior': None}
        # {'Contextual': False, 'Prior': P_Z},
        # {'Contextual': True, 'Prior': P_Z},
    ]

    for experiment in experiments:
        contextualR = experiment['Contextual']
        prior = experiment['Prior']
        # r_sigma = experiment['R_sigma'] #environent noise
        # std=experiment['std'] #Likelihood sigma

        filename = {}
        filename['Samples'] = Samples
        filename['Contextual'] = experiment['Contextual']
        filename['Prior'] = True if (
            experiment['Prior'] is not None) else False
        # filename['Rsigma'] = experiment['R_sigma']
        # filename['std'] = experiment['std']
        print(filename)

        w_stars = {}
        results_div = {}
        results_ttts = {}
        results_lp = {}
        results_greedy = {}

        results_div_correct = {}
        results_ttts_correct = {}
        results_lp_correct = {}
        results_greedy_correct = {}

        results_div_posterior = {}
        results_ttts_posterior = {}
        results_lp_posterior = {}
        results_greedy_posterior = {}

        results_div_err, results_ttts_err, results_lp_err, results_greedy_err = {}, {}, {}, {}
        results_div_posterior_err, results_ttts_posterior_err, results_lp_posterior_err, results_greedy_posterior_err = {}, {}, {}, {}

        _ = list(map(lambda c: collate_results_confidence(c, max_steps, Samples,
                                                          contextualR, prior, sigma, w_stars,
                                                          results_div, results_ttts, results_lp, results_greedy,
                                                          results_div_correct, results_ttts_correct, results_lp_correct, results_greedy_correct,
                                                          results_div_posterior, results_ttts_posterior, results_lp_posterior, results_greedy_posterior,
                                                          results_div_posterior_err, results_ttts_posterior_err, results_lp_posterior_err, results_greedy_posterior_err,
                                                          results_div_err, results_ttts_err, results_lp_err, results_greedy_err), confidences))

        results_groups_correct = [[results_lp_correct[c], results_div_correct[c],
                                   results_ttts_correct[c], results_greedy_correct[c]] for c in confidences]
        print('Correctness: ', results_groups_correct)
        results_groups = [[results_lp[c], results_div[c],
                           results_ttts[c], results_greedy[c]] for c in confidences]
        print('Stop times: ', results_groups)
        results_groups_posterior = [[results_lp_posterior[c], results_div_posterior[c],
                                     results_ttts_posterior[c]] for c in confidences]

        results_groups_err = [[results_lp_err[c], results_div_err[c],
                               results_ttts_err[c], results_greedy_err[c]] for c in confidences]
        print('Stop times: ', results_groups)

        lbs = [w_stars[c] for c in confidences]
        # print("lbs: ", lbs)

        print('Error bars: ', results_groups_err)

        # Plotting

        plot_lines(Samples, contextualR, results_lp_posterior, results_lp_posterior_err, results_div_posterior, results_div_posterior_err,
                   results_ttts_posterior, results_ttts_posterior_err, results_greedy_posterior, results_greedy_posterior_err, lbs)

        ####
        # plot_one_bar(confidences, results_groups, r'$E[\tau]$', r'$1-\delta$',
        #              ['LLPT Explorer', 'Divergence Explorer',
        #                  'TTTS-Latent Explorer', 'Greedy Explorer'],
        #              ylim=[0, 40], opacity=[0.1, 0.1, 1], filename=filename, lbs=lbs, results_groups_err=results_groups_err)
        #
        # plot_one_bar(confidences, results_groups_correct, 'Correctness', r'$1-\delta$',
        #              ['LLPT Explorer', 'Divergence Explorer',
        #                  'TTTS-Latent Explorer', 'Greedy Explorer'],
        #              ylim=[0, 1], opacity=[0.1, 0.1, 1], filename=filename)


def get_means(e, contextualR):
    _ = e.reset()
    x = e.buffer_
    envmodels = e.get_models(e, contextualR=contextualR)

    means = []
    for i in range(6):
        ya, av45, fdg, tau, ptau = envmodels[i].predict(x)
        means.append(ya)

    means = np.array(means)

    """plt.figure()
    for a in range(8):
        gen_data_filtered = gen_data.query(
            'VISCODE == 48 & Z ==0 & A == ' + str(a))[['Y_hat']]
        # print("Z ==3 & A == "+str(a)+" Number of samples: ", len(gen_data_filtered))
        sns.distplot(gen_data_filtered, label='Z=0, A=' + str(a))
    plt.legend(bbox_to_anchor=(1.55, 1.))"""
    #
    #
    return means


def plot_lines(Samples, contextualR, results_lp_posterior, results_lp_posterior_err, results_div_posterior, results_div_posterior_err, results_ttts_posterior, results_ttts_posterior_err, results_greedy_posterior, results_greedy_posterior_err, lbs=None):
    Prior = None
    plt.figure()
    plt.rcParams["figure.figsize"] = (15, 9)
    plt.plot(results_lp_posterior[1], label=r'LLPT Explorer', lw=2)
    # plt.plot(*zip(*sorted(results_lp_posterior.items())), label=r'T&S LP')
    y_err_lp = results_lp_posterior[1] - results_lp_posterior_err[1]
    y__err_lp = results_lp_posterior[1] + results_lp_posterior_err[1]
    y__err_lp = [min(1, i) for i in y__err_lp]
    plt.fill_between(
        range(len(results_lp_posterior[1])), y_err_lp, y__err_lp, alpha=0.1)
    # plt.vlines(np.mean(results_lp[1]), 0, 1, linestyles='--', label='Expected stopping time')

    plt.plot(results_div_posterior[1],
             label=r'Divergence Explorer', lw=2)
    y_err_div = results_div_posterior[1] - results_div_posterior_err[1]
    y__err_div = results_div_posterior[1] + results_div_posterior_err[1]
    y__err_div = [min(1, i) for i in y__err_div]
    plt.fill_between(
        range(len(results_div_posterior[1])), y_err_div, y__err_div, alpha=0.1)
    # plt.vlines(np.mean(results_div[1]), 0, 1, linestyles='--', label='Expected stopping time')

    plt.plot(results_ttts_posterior[1],
             label=r'TTTS-Latent Explorer', lw=2)
    y_err_ttts = results_ttts_posterior[1] - results_ttts_posterior_err[1]
    y__err_ttts = results_ttts_posterior[1] + results_ttts_posterior_err[1]
    y__err_ttts = [min(1, i) for i in y__err_ttts]
    plt.fill_between(
        range(len(results_ttts_posterior[1])), y_err_ttts, y__err_ttts, alpha=0.1)
    # plt.vlines(np.mean(results_ttts[1]), 0, 1, linestyles='--', label='Expected stopping time')

    plt.plot(results_greedy_posterior[1],
             label=r'Greedy Explorer', lw=2)
    y_err_greedy = results_greedy_posterior[1] - \
        results_greedy_posterior_err[1]
    y__err_greedy = results_greedy_posterior[1] + \
        results_greedy_posterior_err[1]
    y__err_greedy = [min(1, i) for i in y__err_greedy]
    plt.fill_between(
        range(len(results_greedy_posterior[1])), y_err_greedy, y__err_greedy, alpha=0.1)
    # plt.vlines(lbs[1]), 0, 1, linestyles = '--', label = 'Expected stopping time')

    # print("lbs: ", lbs)
    if lbs is not None:
        # vlines(x, ymin, ymax, colors='k', linestyles='solid', label=''
        plt.vlines(x=4.503217453131898, ymin=0., ymax=1.5, colors='k', lw=2,
                   linestyles='dashed', label=r'$E[\tau]$ LB for $\delta$=0.01')
        # ax.hlines(lbs, xmin=0., xmax=9, colors='k', linestyles='--', lw=2, label='LB')

    plt.grid(zorder=-100, linestyle='dashed', alpha=0.2)
    plt.ylabel(r'Posterior')  # r'$max(p_t(s))$'
    plt.xlabel(r'$t$')
    plt.xlim(0, 60)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    prior = True if (
        Prior is not None) else False
    plt.savefig('plots/posterior_confidence_Samples_' + str(Samples) + '_Contextual_' + str(contextualR) + '_Prior_' + str(
        prior) + '.pdf', format='pdf', dpi=500)
    plt.show()


def plot_one_bar(confidences, results_groups, y_label, xlabel, xlabels, ylim=[0, 20], opacity=[0.1, 0.1, 1], filename=None, lbs=None, results_groups_err=None):
    # titles, lists
    np.random.seed(0)
    # plt.rcParams["figure.figsize"] = (16, 12)
    # plt.rc('font', size=30, family='serif')
    plt.style.use('tableau-colorblind10')
    # import matplotlib.patches as mpatches
    experiment_typ = None
    w = 0.35    # bar width
    loc = None

    fig, ax = plt.subplots()
    ax.set_ylim(ylim)
    patterns = ["/",  "\\", "+", ".", "+", "\\",
                "x", "o", "O", ".", "|", "+", "*"]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [colors[0], colors[2], colors[1]] + colors[3:]
    plt.ylabel(y_label)
    plt.xlabel(xlabel)

    for idx, alist in enumerate(results_groups):
        # print(confidences)
        # print(alist)
        # lines = plt.plot(alist)

        x = np.linspace(2 * idx, 2 * idx + 1, num=len(alist))
        # print(len(x))
        bars = ax.bar(x,
                      height=alist,
                      # yerr=results_groups[idx],    # error bars
                      capsize=10,  # error bar cap width in points
                      width=w,    # bar width
                      # tick_label=[ti for ti in titles],
                      color=(0, 0, 0, 0),  # face color transparent
                      edgecolor=colors,
                      # hatch=patterns[idx],
                      # ecolor=colors,    # error bar colors; setting this raises an error for whatever reason.
                      label=str(confidences[idx])
                      )

        if(results_groups_err is not None):
            errbar = ax.errorbar(x,
                                 alist,
                                 yerr=np.array(results_groups_err[idx]) / 2.0,
                                 # zorder=-200,
                                 fmt="none",
                                 marker="none",
                                 ecolor='gray',
                                 capsize=10,
                                 alpha=0.7,
                                 elinewidth=2
                                 )

        for i in range(len(bars)):
            bars[i].set(color=colors[i])
            bars[i].set(alpha=1)

    ax.set_xticks([0.5, 2.5, 4.5, 6.5, 8.5])
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in range(len(xlabels)):
        labels[i] = str(xlabels[i])
    ax.set_xticklabels(confidences)

    xlabel = re.sub('\W+', '', xlabel)

    # grid lines
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.2)
    ax.xaxis.grid(color='gray', linestyle='dashed', alpha=0.2)
    loc = 'best'

    if(y_label == r'$E[\tau]$'):
        experiment_typ = 'Stopping_time'

        # ax.ylabel(r'$E[\tau]$')
        if lbs is not None:
            ax.plot([0.5, 2.5, 4.5, 6.5, 8.5], lbs, 'xk--', lw=2, label='LB')
            # ax.hlines(lbs, xmin=0., xmax=9, colors='k', linestyles='--', lw=2, label='LB')

    elif y_label == 'Correctness':
        # loc = "lower right"
        experiment_typ = 'Correctness'
        # ax.ylabel(r'$correctness')
        if xlabel == '1delta':
            ax.plot([0.5, 2.5, 4.5, 6.5, 8.5], [
                    0.7, 0.8, 0.9, 0.95, 0.99], 'xk--', lw=2)
        """else:
            ax.hlines(y=0.95, xmin=0., xmax=9, colors='r',
                      linestyles='--', lw=2, label=r'$\delta$=0.95')"""
    plt.grid(True)

    custom_lines = [Patch(facecolor=colors[i], edgecolor=colors[i],
                          hatch=patterns[i]) for i in range(len(xlabels))]
    legend0 = ax.legend(custom_lines, xlabels,  title="", loc=loc, fontsize=24)
    ax.add_artist(legend0)

    plt.savefig('plots/' + str(experiment_typ) + '_' + str(xlabel) + '_Samples_' + str(filename['Samples']) + '_Contextual_' + str(filename['Contextual']) + '_Prior_' + str(
        filename['Prior']) + '.pdf', format='pdf', dpi=500)  # + '_Rsigma_' + str(filename['Rsigma']) + '_LikelihoodSigma_' + str(filename['std'])
    plt.show()
