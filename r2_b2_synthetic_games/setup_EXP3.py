'''
This script generates and saves the random features required by the EXP3 level-0 strategy.
Run this script before using the EXP3 algorithm as the level-0 strategy
'''

import scipy
import numpy as np
from scipy import special
import pickle

ls = 0.1
def inv_cdf(p):
    return special.erfinv(2 * p - 1) / (np.sqrt(2) * np.pi * ls)

K = 100
x1_domain = np.linspace(0, 1, K)

N_ft = 5

a = np.random.random(N_ft)
ws = inv_cdf(a)

bs = np.random.random(N_ft) * 2 * np.pi

ft_all = []
for x in x1_domain:
    ft_tmp = []
    for i, w in enumerate(ws):
        ft_tmp.append(np.sqrt(2) * np.cos(w * x + bs[i]) / np.sqrt(N_ft))
    ft_all.append(ft_tmp)
features = np.array(ft_all)

pickle.dump(features, open("bandit_features_" + str(N_ft) + ".pkl", "wb"))

