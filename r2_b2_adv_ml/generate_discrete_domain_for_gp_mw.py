"""
This script generates the (uniform) discrete domain that is required by the GP-MW algorithm, for the MNIST experiment.
We assume the dimension of the input action is 2 for both the attacker and defender.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


# K_ represents the dimension of each dimension of the input action of both the attacker and defender
# Since we assume the domain is 2D for each agent (attacker or defender), the cardinality of the domain of each agent is K_ * K_ = 100 * 100
K_ = 100

x1_domain = np.linspace(0, 1, K_)
# print(x1_domain.shape)

sub_domain_player_1 = []
for i in range(K_):
    for j in range(K_):
        sub_domain_player_1.append([x1_domain[i], x1_domain[j]])
sub_domain_player_1 = np.array(sub_domain_player_1)
sub_domain_player_2 = sub_domain_player_1

sub_domains = {"sub_domain_player_1":sub_domain_player_1, "sub_domain_player_2":sub_domain_player_2}
pickle.dump(sub_domains, open("sub_domain_K_10000_D_2.pkl", "wb"))

