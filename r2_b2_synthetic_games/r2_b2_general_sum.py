'''
This script runs the R2-B2 algorithm for two-agent general-sum games using synthetic payoff functions
'''

import GPy
import numpy as np
from bayesian_optimization_r2b2 import BayesianOptimization
import pickle
import os

## load the file containing the randomly generated synthetic payoff functions
file_synth_game = "all_funcs_info_new_K_100_ls_0.1.pkl"
all_funcs_info = pickle.load(open(file_synth_game, "rb"))
domain = all_funcs_info["domain"]
K = all_funcs_info["K"]
length_scale = all_funcs_info["length_scale"]
obs_noise = all_funcs_info["obs_noise"]

# Define the reasoning levels of the two agents
reasoning_level_1 = 0
reasoning_level_2 = 0

# the learning rate parameter used by the GP-MW algorithm
eta = 0.1

# Choose the level-0 mixed strategy among: {"gp_mw", "random", "bandit"},
# which represent the GP-MW, random search, and EXP3 algorithms, respectively.
level_zero_policy = "gp_mw"

# Choose the directory to save the results
log_directory = "results_general_sum_" + level_zero_policy

if not os.path.isdir(log_directory):
    os.mkdir(log_directory)

# whether to use R2-B2-Lite
r2b2_light = False


func_list = np.arange(0, 10)
N_iter = 5 # for each synthetic game, average the results over N_iter=5 random initializations
for i in func_list:
    f1 = all_funcs_info["all_funcs"][i][0]
    f2 = all_funcs_info["all_funcs"][i][1]

    def synth_func_1(param): # payoff function of Agent 1
        ind = np.argmin(np.sum(np.abs(domain - param), axis=1))
        return np.random.normal(f1[ind], obs_noise), f1[ind]

    def synth_func_2(param): # payoff function of Agent 2
        ind = np.argmin(np.sum(np.abs(domain - param), axis=1))
        return np.random.normal(f2[ind], obs_noise), f2[ind]

    for j in range(N_iter):
        # choose the file name to save the results
        if not r2b2_light:
            # if not using the R2-B2-Lite algorithm
            log_file_name = log_directory + "/r2b2_levels_" + str(reasoning_level_1) + "_" + \
                        str(reasoning_level_2) + "_eta_" + str(eta) + "_func_" + str(i) + \
                        "_iter_" + str(j) + "_ls_" + str(length_scale) + ".p"
        else:
            # if using the R2-B2-Lite algorithm
            log_file_name = log_directory + "/r2b2_levels_" + str(reasoning_level_1) + "_" + \
                        str(reasoning_level_2) + "_eta_" + str(eta) + "_func_" + str(i) + \
                        "_iter_" + str(j) + "_ls_" + str(length_scale) + "_r2b2_lite.p"

        lr_BO = BayesianOptimization(f_1=synth_func_1, f_2=synth_func_2,
                pbounds={'x1':(0, 1), 'x2':(0, 1)}, gp_opt_schedule=1000, \
                gp_model='gpy', use_init=None, gp_mcmc=False, \
                log_file=log_file_name, save_init=False, \
                save_init_file=None, fix_gp_hypers=length_scale, domain_size=K*K, \
                level_0_policy_1=level_zero_policy, level_0_policy_2=level_zero_policy, gp_mw_eta=eta, \
                reasoning_level_player_1=reasoning_level_1, \
                reasoning_level_player_2=reasoning_level_2, r2b2_light_player_1=r2b2_light, \
                r2b2_light_player_2=r2b2_light, ARD=False)
        lr_BO.maximize(n_iter=150, init_points=1, kappa=2.0, use_fixed_kappa=False, kappa_scale=0.5, acq='ucb')

        # >> gp_opt_schedule define that the hyperparameters of GP should be updated every gp_opt_schedule iterations; we set it to 1000 since in these synthetic games, we can use the groundtruth GP hyperparamters (fix_gp_hypers=length_scale) and thus no need to update the GP hyperparameters
        # refer to "bayesian_optimization_r2b2.py" for what each of these parameters stand for
