'''
This script generates and saves the payoff functions to be used in all three-types of games (i.e., common-payoff, general-sum and constant-sum games).
Specifically, in all types of two-agent games, the domain of each agent is 1-Dimensional and discrete with the size of K=100.
'''

import GPy
import numpy as np
import pickle


K = 100 # K is the size of the domain for each of the two agent

# the domain of each agent is 1D with size K
x1_domain = np.linspace(0, 1, K)
domain = []
for i in range(K):
    for j in range(K):
        domain.append([x1_domain[i], x1_domain[j]])
domain = np.array(domain)

# In all our synthetic games, we assume the domain of each of the two agents is the same
sub_domain_player_1 = np.linspace(0, 1, K).reshape(-1, 1)
sub_domain_player_2 = np.linspace(0, 1, K).reshape(-1, 1)
sub_domains = {"sub_domain_player_1":sub_domain_player_1, "sub_domain_player_2":sub_domain_player_2}

# "sub_domain_K_100.pkl" saves the discrete domain of each agent, to be used by the algorithm
pickle.dump(sub_domains, open("sub_domain_K_100.pkl", "wb"))

# the length scale of the GP from which the payoff functions are sampled from
length_scale = 0.1

# Since the domain of each agent is 1D and we have two agents, the GP that is used to sample the payoff function should be defined on a 2D domain
kernel = GPy.kern.RBF(input_dim=2, lengthscale=length_scale)
obs_noise = 1e-4

C = kernel.K(domain, domain)
m = np.zeros((C.shape[0])) # We use the mean function which is 0 every where in the domain, following the common practice in GP

func_list = np.arange(0, 10)

# The for loop below generates 10 independent pairs of payoff functions, which corresponds to 10 randomly generated synthetic games
all_funcs = []
for i in func_list:
    print("[generating functions {0}]".format(i))
    # draw a random payoff function and normalize into the range of [0, 1]
    f_1 = np.random.multivariate_normal(m, C, 1).reshape(-1, 1)
    f_1 = (f_1 - np.min(f_1)) / (np.max(f_1) - np.min(f_1))

    f_2 = np.random.multivariate_normal(m, C, 1).reshape(-1, 1)
    f_2 = (f_2 - np.min(f_2)) / (np.max(f_2) - np.min(f_2))

    all_funcs.append([f_1, f_2])
    all_funcs_info = {"all_funcs":all_funcs, "domain":domain, "K":K, "length_scale":length_scale, \
                     "obs_noise":obs_noise}

pickle.dump(all_funcs_info, open("all_funcs_info_new_K_" + str(K) + "_ls_" + str(length_scale) + ".pkl", "wb"))

