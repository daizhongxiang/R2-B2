# -*- coding: utf-8 -*-

# Code adapted based on: https://github.com/fmfn/BayesianOptimization

import numpy as np
import GPy
from helper_funcs_r2b2 import UtilityFunction, unique_rows, PrintLog, acq_max
import pickle
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

class BayesianOptimization(object):

    def __init__(self, f_1, f_2, pbounds, gp_opt_schedule, gp_model, \
                 use_init=False, gp_mcmc=False, log_file=None, save_init=False, save_init_file=None, \
                 fix_gp_hypers=None, ARD=False, domain_size=50, level_0_policy_1="gp_mw", \
                 level_0_policy_2="gp_mw", gp_mw_eta=0.1, reasoning_level_player_1=0,\
                 reasoning_level_player_2=0, r2b2_light_player_1=False, r2b2_light_player_2=False, \
                 sampling_approximation=0, verbose=1):
        """
        f_1 & f_2: payoff function for agent 1 & 2
        pbounds: dictionary defining the search range of each dimension; set it to (0, 1) all the time
        gp_opt_schedule: optimize the GP hyperparameters every "gp_opt_schedule" iterations
        gp_model: use "gpy" all the time
        use_init: whether to use existing initializations
        gp_mcmc: whether to use MCMC to optimize the GP hyperparameters
        log_file: the file in which the results are saved
        save_init: whether to save the initilizations to be used in later runs
        save_init_file: if "save_init" is True, give the file in which the initializations are saved
        fix_gp_hypers: whether to use fixed GP hyperparameters
        ARD: whether to use Automatic Relevance Determination for the GP kernel
        domain_size: size of the joint domain of both agents (not used in the current version)
        level_0_policy_1: the level-0 mixed strategy of Agent 1
        level_0_policy_2: the level-0 mixed strategy of Agent 2
        gp_mw_eta: the learning rate used by the GP-MW level-0 strategy
        reasoning_level_player_1: the reasoning level of Agent 1
        reasoning_level_player_2: the reasoning level of Agent 2
        r2b2_light_player_1: whether to use R2-B2-Lite for Agent 1
        r2b2_light_player_2: whether to use R2-B2-Lite for Agent 2
        sampling_approximation: the number of samples that is used to approximate the expectation in level-1 reasoning
        verbose: whether to print logging information
        """
        
        self.use_init = use_init
        self.ARD = ARD # whether to use ARD for the kernel
        self.fix_gp_hypers = fix_gp_hypers # whether to fix the GP hyperparamters, instead of learning
    
        self.log_file = log_file
        
        self.pbounds = pbounds
        
        self.keys = list(pbounds.keys())
        self.dim = len(pbounds)

        sub_domains = pickle.load(open("sub_domain_K_100.pkl", "rb"))
        self.sub_domain_player_1 = sub_domains["sub_domain_player_1"]
        self.sub_domain_player_2 = sub_domains["sub_domain_player_2"]

        #### we assume the domains of players 1 and 2 have the same size, but this can be easily relaxed
        self.domain_size = self.sub_domain_player_1.shape[0]

        self.dim_player_1 = int(self.dim / 2)
        self.dim_player_2 = int(self.dim / 2)
        self.gp_mw_eta = gp_mw_eta
        self.cum_loss_player_1 = np.zeros(self.domain_size)
        self.cum_loss_player_2 = np.zeros(self.domain_size)
        self.cum_reward_player_1 = np.zeros(self.domain_size)
        self.cum_reward_player_2 = np.zeros(self.domain_size)

        # the level-0 mixed strategy (i.e., distribution over the input domain) of Agent 1 and 2
        self.action_dist_player_1 = np.ones(self.domain_size) / self.domain_size
        self.action_dist_player_2 = np.ones(self.domain_size) / self.domain_size

        ### level_0_policy_player_1 can take the values of "gp_mw", "random" or "bandit"
        self.level_0_policy_player_1 = level_0_policy_1
        self.level_0_policy_player_2 = level_0_policy_2
        
        self.reasoning_level_player_1 = reasoning_level_player_1
        self.reasoning_level_player_2 = reasoning_level_player_2
        
        self.r2b2_light_player_1 = r2b2_light_player_1
        self.r2b2_light_player_2 = r2b2_light_player_2
        
        ### if sampling_approximation == 0, we exactly calculate the expectation when calculating the level-1 action;
        ### otherwise, we approximate the expectation by drawing sampling_approximation samples
        self.sampling_approximation = sampling_approximation

        self.incumbent = None

        
        if self.level_0_policy_player_1 == "bandit" or self.level_0_policy_player_2 == "bandit":
            self.bandit_eta = 0.1
            self.bandit_gamma = 0.1
            self.bandit_N_ft = 5
            self.bandit_features = pickle.load(open("bandit_features_" + str(self.bandit_N_ft) + ".pkl", "rb"))
            self.bandit_cum_loss_player_1 = np.zeros(self.domain_size)
            self.bandit_cum_loss_player_2 = np.zeros(self.domain_size)
            ### use the uniform distribution as the exploration distribution
            self.bandit_action_dist_explore_1 = np.ones(self.domain_size) / self.domain_size
            self.bandit_action_dist_explore_2 = np.ones(self.domain_size) / self.domain_size

        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)

        # The payoff functions to be optimized
        self.f_1 = f_1
        self.f_2 = f_2

        self.initialized = False

        self.init_points = []
        self.x_init = []
        self.y_init = []

        self.X = None
        self.Y_1 = None
        self.Y_2 = None
        self.F_1 = None
        self.F_2 = None

        self.i = 0

        self.gp_mcmc = gp_mcmc
        
        # the gp models as the surrogate functions
        self.gp_model = gp_model
        self.gp_1 = None
        self.gp_params_1 = None
        self.gp_opt_schedule_1 = gp_opt_schedule
        self.gp_2 = None
        self.gp_params_2 = None
        self.gp_opt_schedule_2 = gp_opt_schedule


        self.util = None

        self.plog = PrintLog(self.keys)
        
        self.save_init = save_init
        self.save_init_file = save_init_file

        self.res = {}
        self.res['max'] = {'max_val': None, 'max_params': None}
        self.res['all'] = {'values_1':[], 'values_2':[], 'F_1':[], 'F_2':[], 'params':[], 'init_values':[], \
                           'init_params':[], 'init':[], 'target_f_values':[]}

        self.verbose = verbose
        
        
    def init(self, init_points):
        """
        This function generates init_points random initializations
        """
        
        init_1 = np.random.choice(self.sub_domain_player_1.shape[0], init_points, replace=False)
        init_2 = np.random.choice(self.sub_domain_player_2.shape[0], init_points, replace=False)
        self.X = np.concatenate((self.sub_domain_player_1[init_1, :], self.sub_domain_player_2[init_2, :]), axis=1)

        f_init_1 = []
        f_init_2 = []
        y_init_1 = []
        y_init_2 = []
        for x in self.X:
            curr_y_1, func_1 = self.f_1(x)
            curr_y_2, func_2 = self.f_2(x)

            f_init_1.append(func_1)
            f_init_2.append(func_2)
            y_init_1.append(curr_y_1)
            y_init_2.append(curr_y_2)
            
        self.Y_1 = np.asarray(y_init_1)
        self.Y_2 = np.asarray(y_init_2)
        self.F_1 = np.asarray(f_init_1)
        self.F_2 = np.asarray(f_init_2)

        self.initialized = True

        init = {"X":self.X, "Y_1":self.Y_1, "Y_2":self.Y_2, "F_1":self.F_1, "F_2":self.F_2}
        self.res['all']['init'] = init
        if self.save_init:
            with open("init.p", "wb") as c:
                pickle.dump(init, open(self.save_init_file, "wb"))


    def level_0_policy_player_1_func(self, y_max=1747, iteration=1, gp_samples=None):
        """
        Level-0 reasoning of Agent 1
        y_max: the currently observed max value; not used in the current version
        iteration: the current iteration count
        gp_samples: samples of GP hyperparameters, which are sampled from MCMC; this is also not used in the current version
        """

        if self.level_0_policy_player_1 == "gp_mw":
            ## get the sub_domain with all the input actions whose agent-2 input is equal to x_{2, t-1}
            x_opponent_prev = self.X[-1, self.dim_player_1:]
            rep = np.tile(x_opponent_prev, (self.sub_domain_player_1.shape[0], 1))
            sub_domain = np.concatenate((self.sub_domain_player_1, rep), axis=1)

            para_dict = {"gp":self.gp_1, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
            all_ucb = []
            for x in sub_domain:
                all_ucb.append(-self.util.utility(x, para_dict)[0][0])
            self.cum_reward_player_1 += np.array(all_ucb)

            dist_un_norm = np.exp(self.gp_mw_eta * self.cum_reward_player_1)
            self.action_dist_player_1 = dist_un_norm / np.sum(dist_un_norm)

            domain_ind = np.arange(sub_domain.shape[0])
            x_1_ind = np.random.choice(domain_ind, 1, p=self.action_dist_player_1)
            x_1 = sub_domain[x_1_ind, :self.dim_player_1]

        elif self.level_0_policy_player_1 == "random":
            self.action_dist_player_1 = np.ones(self.sub_domain_player_1.shape[0]) / self.sub_domain_player_1.shape[0]
            rand_ind = np.random.choice(self.sub_domain_player_1.shape[0], 1)
            x_1 = self.sub_domain_player_1[rand_ind, :]

        elif self.level_0_policy_player_1 == "bandit":
            x_prev = self.X[-1, :self.dim_player_1]
            ind = np.argmin(np.sum(np.abs(self.sub_domain_player_1 - x_prev), axis=1))
            a_prev = self.bandit_features[ind, :]
            y_prev = -self.Y_1[-1]

            Q_t = np.zeros((self.bandit_N_ft, self.bandit_N_ft))
            for i, a in enumerate(self.bandit_features):
                Q_t += np.outer(a, a) * self.action_dist_player_1[i]
            Q_t_inv = np.linalg.inv(Q_t)
            Y_t_cap = Q_t_inv.dot(a_prev) * y_prev

            loss_est = np.zeros(self.bandit_features.shape[0])
            for i, a in enumerate(self.bandit_features):
                loss_est[i] = a.dot(Y_t_cap)

            self.bandit_cum_loss_player_1 += loss_est
            dist_un_norm = np.exp(-self.bandit_eta * self.bandit_cum_loss_player_1)
            bandit_action_dist_player_1_tmp = dist_un_norm / np.sum(dist_un_norm)
        
            self.action_dist_player_1 = self.bandit_gamma * self.bandit_action_dist_explore_1 + \
                        (1 - self.bandit_gamma) * bandit_action_dist_player_1_tmp

            domain_ind = np.arange(self.bandit_features.shape[0])
            x_1_ind = np.random.choice(domain_ind, 1, p=self.action_dist_player_1)
            x_1 = self.sub_domain_player_1[x_1_ind, :]

        return x_1


    def level_0_policy_player_2_func(self, y_max=1747, iteration=1, gp_samples=None):
        """
        Level-0 reasoning of Agent 2
        """
        if self.level_0_policy_player_2 == "gp_mw":
            x_opponent_prev = self.X[-1, :self.dim_player_1]
            rep = np.tile(x_opponent_prev, (self.sub_domain_player_2.shape[0], 1))
            sub_domain = np.concatenate((rep, self.sub_domain_player_2), axis=1)

            para_dict = {"gp":self.gp_2, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
            all_ucb = []
            for x in sub_domain:
                all_ucb.append(-self.util.utility(x, para_dict)[0][0])
            self.cum_reward_player_2 += np.array(all_ucb)

            dist_un_norm = np.exp(self.gp_mw_eta * self.cum_reward_player_2)
            self.action_dist_player_2 = dist_un_norm / np.sum(dist_un_norm)

            domain_ind = np.arange(sub_domain.shape[0])
            x_2_ind = np.random.choice(domain_ind, 1, p=self.action_dist_player_2)
            x_2 = sub_domain[x_2_ind, self.dim_player_1:]

        elif self.level_0_policy_player_2 == "random":
            self.action_dist_player_2 = \
                np.ones(self.sub_domain_player_2.shape[0]) / self.sub_domain_player_2.shape[0]
            rand_ind = np.random.choice(self.sub_domain_player_2.shape[0], 1)
            x_2 = self.sub_domain_player_2[rand_ind, :]
    
        elif self.level_0_policy_player_2 == "bandit":
            x_prev = self.X[-1, self.dim_player_1:]
            ind = np.argmin(np.sum(np.abs(self.sub_domain_player_2 - x_prev), axis=1))
            a_prev = self.bandit_features[ind, :]
            y_prev = -self.Y_2[-1]

            Q_t = np.zeros((self.bandit_N_ft, self.bandit_N_ft))
            for i, a in enumerate(self.bandit_features):
                Q_t += np.outer(a, a) * self.action_dist_player_2[i]
            Q_t_inv = np.linalg.inv(Q_t)
            Y_t_cap = Q_t_inv.dot(a_prev) * y_prev

            loss_est = np.zeros(self.bandit_features.shape[0])
            for i, a in enumerate(self.bandit_features):
                loss_est[i] = a.dot(Y_t_cap)

            self.bandit_cum_loss_player_2 += loss_est
            dist_un_norm = np.exp(-self.bandit_eta * self.bandit_cum_loss_player_2)
            bandit_action_dist_player_2_tmp = dist_un_norm / np.sum(dist_un_norm)

            self.action_dist_player_2 = self.bandit_gamma * self.bandit_action_dist_explore_2 + \
                        (1 - self.bandit_gamma) * bandit_action_dist_player_2_tmp

            domain_ind = np.arange(self.bandit_features.shape[0])
            x_2_ind = np.random.choice(domain_ind, 1, p=self.action_dist_player_2)
            x_2 = self.sub_domain_player_2[x_2_ind, :]
    
        return x_2


    def level_1_policy_player_1_func(self, y_max=1747, iteration=1, gp_samples=None):
        """
        Level-1 reasoning of Agent 1
        """
        if not self.r2b2_light_player_1:
            # if R2-B2-Lite is not taken
            x_1, all_ucb = acq_max(ac=self.util_rr.utility, gp=self.gp_1, y_max=y_max, bounds=self.bounds, \
                    iteration=iteration, gp_samples=gp_samples, \
                    player_id=1, action_dist=self.action_dist_player_2, \
                    sub_domain_player_1=self.sub_domain_player_1, \
                    sub_domain_player_2=self.sub_domain_player_2, \
                    sampling_approximation=self.sampling_approximation)
            x_1 = x_1.reshape(1, -1)

        else:
            # if R2-B2-Lite is taken
            print("[R2-B2_Lite for Player 1]")
            domain_ind = np.arange(self.sub_domain_player_2.shape[0])
            x_2_ind = np.random.choice(domain_ind, 1, p=self.action_dist_player_2)
            x_2_sim = self.sub_domain_player_2[x_2_ind, :]

            rep = np.tile(x_2_sim, (self.sub_domain_player_1.shape[0], 1))
            sub_domain = np.concatenate((self.sub_domain_player_1, rep), axis=1)

            para_dict = {"gp":self.gp_1, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
            all_ucb = []
            for x in sub_domain:
                all_ucb.append(-self.util.utility(x, para_dict)[0][0])
            all_ucb = np.array(all_ucb)
            x_1 = self.sub_domain_player_1[np.argmax(all_ucb)]
            x_1 = x_1.reshape(1, -1)

        return x_1, all_ucb


    def level_1_policy_player_2_func(self, y_max=1747, iteration=1, gp_samples=None):
        """
        Level-1 reasoning of Agent 2
        """
        if not self.r2b2_light_player_2:
            # if R2-B2-Lite is not taken
            x_2, all_ucb = acq_max(ac=self.util_rr.utility, gp=self.gp_2, y_max=y_max, bounds=self.bounds, \
                    iteration=iteration, gp_samples=gp_samples, \
                    player_id=2, action_dist=self.action_dist_player_1, \
                    sub_domain_player_1=self.sub_domain_player_1, \
                    sub_domain_player_2=self.sub_domain_player_2, \
                    sampling_approximation=self.sampling_approximation)
            x_2 = x_2.reshape(1, -1)
        else:
            # if R2-B2-Lite is taken
            print("[R2-B2_Lite for Player 2]")
            domain_ind = np.arange(self.sub_domain_player_1.shape[0])
            x_1_ind = np.random.choice(domain_ind, 1, p=self.action_dist_player_1)
            x_1_sim = self.sub_domain_player_1[x_1_ind, :]
            
            rep = np.tile(x_1_sim, (self.sub_domain_player_2.shape[0], 1))
            sub_domain = np.concatenate((rep, self.sub_domain_player_2), axis=1)

            para_dict = {"gp":self.gp_2, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
            all_ucb = []
            for x in sub_domain:
                all_ucb.append(-self.util.utility(x, para_dict)[0][0])
            all_ucb = np.array(all_ucb)
            x_2 = self.sub_domain_player_2[np.argmax(all_ucb)]
            x_2 = x_2.reshape(1, -1)

        return x_2, all_ucb

    def level_k_policy_player_1_func(self, y_max=1747, iteration=1, gp_samples=None):
        """
        Level-k>1 reasoning of Agent 1
        """

        if self.reasoning_level_player_1 % 2 == 0: 
            # if the reasoning level is an even number, we need to start by calculating the level-1 policy of the opponent (Agent 2)
            k = 2
            
            x_2, _ = self.level_1_policy_player_2_func(iteration=iteration)

            #### Agent 1 best-responds
            x_2 = np.squeeze(x_2)
            rep = np.tile(x_2, (self.sub_domain_player_1.shape[0], 1))
            sub_domain = np.concatenate((self.sub_domain_player_1, rep), axis=1)
            para_dict = {"gp":self.gp_1, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
            all_ucb = []
            for x in sub_domain:
                all_ucb.append(-self.util.utility(x, para_dict)[0][0])
            all_ucb = np.array(all_ucb)
            x_1 = self.sub_domain_player_1[np.argmax(all_ucb)]
            x_1 = x_1.reshape(1, -1)

            while self.reasoning_level_player_1 > k:
                #### Agent 2 best-responds
                x_1 = np.squeeze(x_1)
                rep = np.tile(x_1, (self.sub_domain_player_2.shape[0], 1))
                sub_domain = np.concatenate((rep, self.sub_domain_player_2), axis=1)
                para_dict = {"gp":self.gp_2, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
                all_ucb = []
                for x in sub_domain:
                    all_ucb.append(-self.util.utility(x, para_dict)[0][0])
                all_ucb = np.array(all_ucb)
                x_2 = self.sub_domain_player_2[np.argmax(all_ucb)]
                x_2 = x_2.reshape(1, -1)

                #### Agent 1 best-responds
                x_2 = np.squeeze(x_2)
                rep = np.tile(x_2, (self.sub_domain_player_1.shape[0], 1))
                sub_domain = np.concatenate((self.sub_domain_player_1, rep), axis=1)
                para_dict = {"gp":self.gp_1, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
                all_ucb = []
                for x in sub_domain:
                    all_ucb.append(-self.util.utility(x, para_dict)[0][0])
                all_ucb = np.array(all_ucb)
                x_1 = self.sub_domain_player_1[np.argmax(all_ucb)]
                x_1 = x_1.reshape(1, -1)
                
                k += 2

        else:
            # if the reasoning level is an odd number, we need to start by calculating the level-1 policy of Agent 1
            k = 3

            x_1, _ = self.level_1_policy_player_1_func(iteration=iteration)
            
            #### Agent 2 best-responds
            x_1 = np.squeeze(x_1)
            rep = np.tile(x_1, (self.sub_domain_player_2.shape[0], 1))
            sub_domain = np.concatenate((rep, self.sub_domain_player_2), axis=1)
            para_dict = {"gp":self.gp_2, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
            all_ucb = []
            for x in sub_domain:
                all_ucb.append(-self.util.utility(x, para_dict)[0][0])
            all_ucb = np.array(all_ucb)
            x_2 = self.sub_domain_player_2[np.argmax(all_ucb)]
            x_2 = x_2.reshape(1, -1)

            #### Agent 1 best-responds
            x_2 = np.squeeze(x_2)
            rep = np.tile(x_2, (self.sub_domain_player_1.shape[0], 1))
            sub_domain = np.concatenate((self.sub_domain_player_1, rep), axis=1)
            para_dict = {"gp":self.gp_1, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
            all_ucb = []
            for x in sub_domain:
                all_ucb.append(-self.util.utility(x, para_dict)[0][0])
            all_ucb = np.array(all_ucb)
            x_1 = self.sub_domain_player_1[np.argmax(all_ucb)]
            x_1 = x_1.reshape(1, -1)
            
            while self.reasoning_level_player_1 > k:
                #### Agent 2 best-responds
                x_1 = np.squeeze(x_1)
                rep = np.tile(x_1, (self.sub_domain_player_2.shape[0], 1))
                sub_domain = np.concatenate((rep, self.sub_domain_player_2), axis=1)
                para_dict = {"gp":self.gp_2, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
                all_ucb = []
                for x in sub_domain:
                    all_ucb.append(-self.util.utility(x, para_dict)[0][0])
                all_ucb = np.array(all_ucb)
                x_2 = self.sub_domain_player_2[np.argmax(all_ucb)]
                x_2 = x_2.reshape(1, -1)

                #### Agent 1 best-responds
                x_2 = np.squeeze(x_2)
                rep = np.tile(x_2, (self.sub_domain_player_1.shape[0], 1))
                sub_domain = np.concatenate((self.sub_domain_player_1, rep), axis=1)
                para_dict = {"gp":self.gp_1, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
                all_ucb = []
                for x in sub_domain:
                    all_ucb.append(-self.util.utility(x, para_dict)[0][0])
                all_ucb = np.array(all_ucb)
                x_1 = self.sub_domain_player_1[np.argmax(all_ucb)]
                x_1 = x_1.reshape(1, -1)
                
                k += 2

        return x_1, all_ucb

    def level_k_policy_player_2_func(self, y_max=1747, iteration=1, gp_samples=None):
        """
        Level-k>1 reasoning of Agent 2
        """
        if self.reasoning_level_player_2 % 2 == 0: 
            # if the reasoning level is an even number, we need to start by calculating the level-1 policy of the opponent (Agent 1)
            k = 2
            
            x_1, _ = self.level_1_policy_player_1_func(iteration=iteration)
            
            #### Agent 2 best-responds
            x_1 = np.squeeze(x_1)
            rep = np.tile(x_1, (self.sub_domain_player_2.shape[0], 1))
            sub_domain = np.concatenate((rep, self.sub_domain_player_1), axis=1)
            para_dict = {"gp":self.gp_2, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
            all_ucb = []
            for x in sub_domain:
                all_ucb.append(-self.util.utility(x, para_dict)[0][0])
            all_ucb = np.array(all_ucb)
            x_2 = self.sub_domain_player_2[np.argmax(all_ucb)]
            x_2 = x_2.reshape(1, -1)
    
            while self.reasoning_level_player_2 > k:
                #### Agent 1 best-responds
                x_2 = np.squeeze(x_2)
                rep = np.tile(x_2, (self.sub_domain_player_1.shape[0], 1))
                sub_domain = np.concatenate((self.sub_domain_player_1, rep), axis=1)
                para_dict = {"gp":self.gp_1, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
                all_ucb = []
                for x in sub_domain:
                    all_ucb.append(-self.util.utility(x, para_dict)[0][0])
                all_ucb = np.array(all_ucb)
                x_1 = self.sub_domain_player_1[np.argmax(all_ucb)]
                x_1 = x_1.reshape(1, -1)

                #### Agent 2 best-responds
                x_1 = np.squeeze(x_1)
                rep = np.tile(x_1, (self.sub_domain_player_2.shape[0], 1))
                sub_domain = np.concatenate((rep, self.sub_domain_player_2), axis=1)
                para_dict = {"gp":self.gp_2, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
                all_ucb = []
                for x in sub_domain:
                    all_ucb.append(-self.util.utility(x, para_dict)[0][0])
                all_ucb = np.array(all_ucb)
                x_2 = self.sub_domain_player_2[np.argmax(all_ucb)]
                x_2 = x_2.reshape(1, -1)
                
                k += 2

        else: 
            # if the reasoning level is an odd number, we need to start by calculating the level-1 policy of Agent 2
            k = 3
            
            x_2, _ = self.level_1_policy_player_2_func(iteration=iteration)
            
            #### Agent 1 best-responds
            x_2 = np.squeeze(x_2)
            rep = np.tile(x_2, (self.sub_domain_player_1.shape[0], 1))
            sub_domain = np.concatenate((self.sub_domain_player_1, rep), axis=1)
            para_dict = {"gp":self.gp_1, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
            all_ucb = []
            for x in sub_domain:
                all_ucb.append(-self.util.utility(x, para_dict)[0][0])
            all_ucb = np.array(all_ucb)
            x_1 = self.sub_domain_player_1[np.argmax(all_ucb)]
            x_1 = x_1.reshape(1, -1)

            #### Agent 2 best-responds
            x_1 = np.squeeze(x_1)
            rep = np.tile(x_1, (self.sub_domain_player_2.shape[0], 1))
            sub_domain = np.concatenate((rep, self.sub_domain_player_2), axis=1)
            para_dict = {"gp":self.gp_2, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
            all_ucb = []
            for x in sub_domain:
                all_ucb.append(-self.util.utility(x, para_dict)[0][0])
            all_ucb = np.array(all_ucb)
            x_2 = self.sub_domain_player_2[np.argmax(all_ucb)]
            x_2 = x_2.reshape(1, -1)
            
            while self.reasoning_level_player_2 > k:
                #### Agent 1 best-responds
                x_2 = np.squeeze(x_2)
                rep = np.tile(x_2, (self.sub_domain_player_1.shape[0], 1))
                sub_domain = np.concatenate((self.sub_domain_player_1, rep), axis=1)
                para_dict = {"gp":self.gp_1, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
                all_ucb = []
                for x in sub_domain:
                    all_ucb.append(-self.util.utility(x, para_dict)[0][0])
                all_ucb = np.array(all_ucb)
                x_1 = self.sub_domain_player_1[np.argmax(all_ucb)]
                x_1 = x_1.reshape(1, -1)

                #### Agent 2 best-responds
                x_1 = np.squeeze(x_1)
                rep = np.tile(x_1, (self.sub_domain_player_2.shape[0], 1))
                sub_domain = np.concatenate((rep, self.sub_domain_player_2), axis=1)
                para_dict = {"gp":self.gp_2, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples}
                all_ucb = []
                for x in sub_domain:
                    all_ucb.append(-self.util.utility(x, para_dict)[0][0])
                all_ucb = np.array(all_ucb)
                x_2 = self.sub_domain_player_2[np.argmax(all_ucb)]
                x_2 = x_2.reshape(1, -1)
                
                k += 2

        return x_2, all_ucb
    
    
    
    def maximize(self,
                 init_points=5,
                 n_iter=100,
                 acq='ucb',
                 kappa=2.0,
                 use_fixed_kappa=False,
                 kappa_scale=0.2,
                 xi=0.0,):

        self.plog.reset_timer()

        self.util = UtilityFunction(kind="ucb", kappa=kappa, use_fixed_kappa=use_fixed_kappa, kappa_scale=kappa_scale, xi=xi, gp_model=self.gp_model)
        self.util_rr = UtilityFunction(kind="ucb_rr", kappa=kappa, use_fixed_kappa=use_fixed_kappa, kappa_scale=kappa_scale, xi=xi, gp_model=self.gp_model)

        
        # Initialization
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()

            if self.use_init != None:
                init = pickle.load(open(self.use_init, "rb"))

                self.X, self.Y_1, self.Y_2, self.F_1, self.F_2 = init["X"], init["Y_1"], init["Y_2"], init["F_1"], init["F_2"]
                self.initialized = True
                self.res['all']['init'] = init
            else:
                self.init(init_points)

        y_max = self.Y_1.max() # this is a placeholder, since y_max is not used in the current version

        # Find the unique rows (i.e., non-repeated inputs) of X to prevent error in GP inference
        ur = unique_rows(self.X)
        if self.fix_gp_hypers is None:
            self.gp_1 = GPy.models.GPRegression(self.X[ur], self.Y_1[ur].reshape(-1, 1), \
                    GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=0.05, ARD=self.ARD))
            self.gp_2 = GPy.models.GPRegression(self.X[ur], self.Y_2[ur].reshape(-1, 1), \
                    GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=0.05, ARD=self.ARD))

            if self.gp_mcmc:
                self.gp_1.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                self.gp_1.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                self.gp_1.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                print("[Running MCMC for GP hyper-parameters]")
                hmc = GPy.inference.mcmc.HMC(self.gp_1, stepsize=5e-2)
                gp_samples = hmc.sample(num_samples=500)[-300:] # Burnin
                gp_samples_mean = np.mean(gp_samples, axis=0)
                print("Mean of MCMC hypers: {0}".format(gp_samples_mean))

                self.gp_1.kern.variance.fix(gp_samples_mean[0])
                self.gp_1.kern.lengthscale.fix(gp_samples_mean[1])
                self.gp_1.likelihood.variance.fix(gp_samples_mean[2])

                self.gp_2.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                self.gp_2.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                self.gp_2.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                print("[Running MCMC for GP hyper-parameters]")
                hmc = GPy.inference.mcmc.HMC(self.gp_2, stepsize=5e-2)
                gp_samples = hmc.sample(num_samples=500)[-300:] # Burnin
                gp_samples_mean = np.mean(gp_samples, axis=0)
                print("Mean of MCMC hypers: {0}".format(gp_samples_mean))

                self.gp_2.kern.variance.fix(gp_samples_mean[0])
                self.gp_2.kern.lengthscale.fix(gp_samples_mean[1])
                self.gp_2.likelihood.variance.fix(gp_samples_mean[2])

            else:
                if self.fix_gp_hypers is None:
                    self.gp_1.optimize_restarts(num_restarts = 10, messages=False)
                    self.gp_params_1 = self.gp_1.parameters
                gp_samples = None # set this flag variable to None, to indicate that MCMC is not used
                print("---Optimized hyper: ", self.gp_1)

                if self.fix_gp_hypers is None:
                    self.gp_2.optimize_restarts(num_restarts = 10, messages=False)
                    self.gp_params_2 = self.gp_2.parameters
                gp_samples = None # set this flag variable to None, to indicate that MCMC is not used
                print("---Optimized hyper: ", self.gp_2)

        else:
            self.gp_1 = GPy.models.GPRegression(self.X[ur], self.Y_1[ur].reshape(-1, 1), \
                    GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=self.fix_gp_hypers, ARD=self.ARD))
            self.gp_2 = GPy.models.GPRegression(self.X[ur], self.Y_2[ur].reshape(-1, 1), \
                    GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=self.fix_gp_hypers, ARD=self.ARD))


        print("[Agent 1 reasons at level {0}]".format(self.reasoning_level_player_1))
        print("[Agent 2 reasons at level {0}]".format(self.reasoning_level_player_2))


        x_1 = self.level_0_policy_player_1_func(iteration=1)
        x_2 = self.level_0_policy_player_2_func(iteration=1)

        if self.reasoning_level_player_1 == 1: 
            x_1, all_ucb_1 = self.level_1_policy_player_1_func(iteration=1)
        elif self.reasoning_level_player_1 > 1:
            x_1, all_ucb_1 = self.level_k_policy_player_1_func(iteration=1)

        if self.reasoning_level_player_2 == 1:
            x_2, all_ucb_2 = self.level_1_policy_player_2_func(iteration=1)
        elif self.reasoning_level_player_2 > 1:
            x_2, all_ucb_2 = self.level_k_policy_player_2_func(iteration=1)

        x_max = np.squeeze(np.concatenate((x_1, x_2), axis=1))


        if self.verbose:
            self.plog.print_header(initialization=False)

        for i in range(n_iter):
            repeat_flag = False
            if np.any(np.all(self.X - x_max == 0, axis=1)):
                # This handles the case in which the joint action selected by both agents (x_max) has been queried before;
                # in particular, when this happens, the same joint action is queried, and the new observation is used to replace the corresponding old observation; i.e., no new input-output pair is added to the data of the GP-surrogate
                repeat_flag = True
                rep_ind = np.argmin(np.sum(np.abs(self.X - x_max), axis=1))

            curr_y_1, func_1 = self.f_1(x_max)
            curr_y_2, func_2 = self.f_2(x_max)

            x_max_prev = x_max.copy()
            if not repeat_flag:
                # if the joint action is not a repeated query
                self.F_1 = np.append(self.F_1, func_1)
                self.F_2 = np.append(self.F_2, func_2)
                self.Y_1 = np.append(self.Y_1, curr_y_1)
                self.Y_2 = np.append(self.Y_2, curr_y_2)
                self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            else:
                # if the joint action is a repeated query, we replace the corresponding old observation with the newly observed one
                self.Y_1[rep_ind] = curr_y_1
                self.Y_2[rep_ind] = curr_y_2

            if self.verbose:
                self.plog.print_step(x_max, self.Y_1[-1], warning=False)

            if self.Y_1[-1] > y_max:
                y_max = self.Y_1[-1]
                self.incumbent = self.Y_1[-1]

            ur = unique_rows(self.X)

            self.gp_1.set_XY(X=self.X[ur], Y=self.Y_1[ur].reshape(-1, 1))
            self.gp_2.set_XY(X=self.X[ur], Y=self.Y_2[ur].reshape(-1, 1))

            if i >= self.gp_opt_schedule_1 and i % self.gp_opt_schedule_1 == 0:
                if self.gp_mcmc:
                    self.gp_1.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                    self.gp_1.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                    self.gp_1.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                    print("[Running MCMC for GP hyper-parameters]")
                    hmc = GPy.inference.mcmc.HMC(self.gp_1, stepsize=5e-2)
                    gp_samples = hmc.sample(num_samples=500)[-300:] # Burnin

                    gp_samples_mean = np.mean(gp_samples, axis=0)
                    print("Mean of MCMC hypers: {0}".format(gp_samples_mean))

                    self.gp_1.kern.variance.fix(gp_samples_mean[0])
                    self.gp_1.kern.lengthscale.fix(gp_samples_mean[1])
                    self.gp_1.likelihood.variance.fix(gp_samples_mean[2])


                    self.gp_2.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                    self.gp_2.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                    self.gp_2.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                    print("[Running MCMC for GP hyper-parameters]")
                    hmc = GPy.inference.mcmc.HMC(self.gp_2, stepsize=5e-2)
                    gp_samples = hmc.sample(num_samples=500)[-300:] # Burnin

                    gp_samples_mean = np.mean(gp_samples, axis=0)
                    print("Mean of MCMC hypers: {0}".format(gp_samples_mean))

                    self.gp_2.kern.variance.fix(gp_samples_mean[0])
                    self.gp_2.kern.lengthscale.fix(gp_samples_mean[1])
                    self.gp_2.likelihood.variance.fix(gp_samples_mean[2])

                else:
                    self.gp_1.optimize_restarts(num_restarts = 10, messages=False)
                    self.gp_params_1 = self.gp_1.parameters
                    gp_samples = None # set this flag variable to None, to indicate that MCMC is not used
                    print("---Optimized hyper: ", self.gp_1)

                    self.gp_2.optimize_restarts(num_restarts = 10, messages=False)
                    self.gp_params_2 = self.gp_2.parameters
                    gp_samples = None # set this flag variable to None, to indicate that MCMC is not used
                    print("---Optimized hyper: ", self.gp_2)

            x_1 = self.level_0_policy_player_1_func(iteration=i+2)
            x_2 = self.level_0_policy_player_2_func(iteration=i+2)
            
            if self.reasoning_level_player_1 == 1:
                x_1, all_ucb_1 = self.level_1_policy_player_1_func(iteration=i+2)
            elif self.reasoning_level_player_1 > 1:
                x_1, all_ucb_1 = self.level_k_policy_player_1_func(iteration=i+2)

            if self.reasoning_level_player_2 == 1:
                x_2, all_ucb_2 = self.level_1_policy_player_2_func(iteration=i+2)
            elif self.reasoning_level_player_2 > 1:
                x_2, all_ucb_2 = self.level_k_policy_player_2_func(iteration=i+2)

            x_max = np.squeeze(np.concatenate((x_1, x_2), axis=1))

            self.i += 1

            if not repeat_flag: # if it's not a repeated query
                # if the joint action is not a repeated query
                self.res['all']['F_1'].append(self.F_1[-1])
                self.res['all']['F_2'].append(self.F_2[-1])
                self.res['all']['values_1'].append(self.Y_1[-1])
                self.res['all']['values_2'].append(self.Y_2[-1])
                self.res['all']['params'].append(self.X[-1])
            else:
                # if the joint action is a repeated query, we replace the corresponding old observation with the newly observed one
                self.res['all']['F_1'].append(func_1[0])
                self.res['all']['F_2'].append(func_2[0])
                self.res['all']['values_1'].append(curr_y_1[0])
                self.res['all']['values_2'].append(curr_y_2[0])
                self.res['all']['params'].append(x_max_prev)

        if self.log_file is not None:
            pickle.dump(self.res, open(self.log_file, "wb"))

        if self.verbose:
            self.plog.print_summary()
