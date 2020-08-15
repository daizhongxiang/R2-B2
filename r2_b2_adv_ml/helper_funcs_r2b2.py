# Code adapted based on: https://github.com/fmfn/BayesianOptimization

#from __future__ import print_function
#from __future__ import division
import numpy as np
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipydirect import minimize as mini_direct
import pickle
USE_DIRECT_OPTIMIZER = True

def acq_max(ac, gp, y_max, bounds, iteration, gp_samples, player_id, approx_samples, dim_1, dim_2, sub_domain, action_dist):
    if not USE_DIRECT_OPTIMIZER:
        # so we assume we always us the DIRECR optimization algorithm to optimize the acquisition function
        pass
    else:
        print("[Running the DIRECT optimizer]")
        
        if sub_domain is not None:
            # if the domain is discrete/finite, we simply enumerate the acquisition function value over the entire domain
            
            para_dict={"gp":gp, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples, "player_id":player_id, \
                       "approx_samples":approx_samples, "sub_domain":sub_domain, "action_dist":action_dist}

            ys = []
            for x in tqdm(sub_domain):
                ys.append(ac(x, para_dict))

            ys = np.squeeze(np.array(ys))
            argmin_ind = np.argmin(ys)
            
            x_max = sub_domain[argmin_ind, :]
        else:
            # if the domain is continuous, we use the DIRECT algorithm to optimize the acquisition function
            bound_list = []
            for b in bounds:
                bound_list.append(tuple(b))

            if player_id == 1:
                bound_list = bound_list[:dim_1]
            else:
                bound_list = bound_list[dim_1:]

            res = mini_direct(ac, bound_list, para_dict={"gp":gp, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples, "player_id":player_id, "approx_samples":approx_samples, "sub_domain":sub_domain, "action_dist":action_dist})
            x_max = res["x"]

    return x_max, None

def acq_max_determ(ac, gp, y_max, bounds, iteration, gp_samples, player_id, x_sim, dim_1, dim_2, sub_domain):
    if not USE_DIRECT_OPTIMIZER:
        pass
    else:
        print("[Running the DIRECT optimizer]")
        if sub_domain is not None:
            # if the domain is discrete/finite, we simply enumerate the acquisition function value over the entire domain
            para_dict={"gp":gp, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples, "player_id":player_id, "x_sim":x_sim}

            ys = []
            for x in tqdm(sub_domain):
                ys.append(ac(x, para_dict))

            ys = np.squeeze(np.array(ys))
            argmin_ind = np.argmin(ys)
            
            x_max = sub_domain[argmin_ind, :]
        else:   
            # if the domain is continuous, we use the DIRECT algorithm to optimize the acquisition function

            bound_list = []
            for b in bounds:
                bound_list.append(tuple(b))

            if player_id == 1:
                bound_list = bound_list[:dim_1]
            else:
                bound_list = bound_list[dim_1:]

            res = mini_direct(ac, bound_list, para_dict={"gp":gp, "y_max":y_max, "iteration":iteration, "gp_samples":gp_samples, "player_id":player_id, "x_sim":x_sim})
            x_max = res["x"]

    return x_max, None

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, use_fixed_kappa, kappa_scale, xi, gp_model):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa
        self.use_fixed_kappa = use_fixed_kappa
        self.kappa_scale = kappa_scale
        
        self.xi = xi
        self.gp_model = gp_model

        if kind not in ['ucb', 'ucb_rr']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb or ucb_rr.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    # This function is defined to work with the DIRECT optimizer
    def utility(self, x, para_dict):
        if self.kind == 'ucb':
            gp, y_max, iteration, gp_samples, player_id, x_sim = \
                    para_dict["gp"], para_dict["y_max"], \
                    para_dict["iteration"], para_dict["gp_samples"], para_dict["player_id"], \
                    para_dict["x_sim"]
        elif self.kind == 'ucb_rr': # for now the other case is "ucb_rr"
            gp, y_max, iteration, gp_samples, player_id, approx_samples, sub_domain, action_dist = \
                    para_dict["gp"], para_dict["y_max"], \
                    para_dict["iteration"], para_dict["gp_samples"], para_dict["player_id"], \
                    para_dict["approx_samples"], para_dict["sub_domain"], para_dict["action_dist"]

        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa, self.use_fixed_kappa, self.kappa_scale, iteration, \
                             self.gp_model, gp_samples, player_id, x_sim)
        if self.kind == 'ucb_rr':
            return self._ucb_rr(x, gp, self.kappa, self.use_fixed_kappa, self.kappa_scale, iteration, \
                             self.gp_model, gp_samples, player_id, approx_samples, sub_domain, action_dist)

    @staticmethod
    def _ucb_rr(x, gp, kappa, use_fixed_kappa, kappa_scale, iteration, gp_model, gp_samples, player_id, approx_samples, sub_domain, action_dist):
        if USE_DIRECT_OPTIMIZER:
            x = x.reshape(1, -1)
        d = x.shape[1]

        if approx_samples == []:
            all_x_comb = []
            for i in range(len(action_dist)):
                if player_id == 1:
                    x_comb = np.concatenate((x, sub_domain[i, :].reshape(1, -1)), axis=1)
                elif player_id == 2:
                    x_comb = np.concatenate((sub_domain[i, :].reshape(1, -1), x), axis=1)
                all_x_comb.append(x_comb)
            all_x_comb = np.squeeze(np.array(all_x_comb))

            mean, var = gp.predict(all_x_comb)
            std = np.sqrt(var)

            if use_fixed_kappa:
                ucb_tmp = mean + np.sqrt(kappa) * std
            else:
                ucb_tmp = mean + np.sqrt(kappa_scale * d * np.log(2 * iteration)) * std

            ucb_cum = np.sum(np.multiply(ucb_tmp, action_dist.reshape(-1, 1)))

        else:
            ## if appr_samples != [], we approximately calculate the expectation using the sampled indices
            all_x_comb = []
            rep = np.tile(x, (approx_samples.shape[0], 1))
            if player_id == 1:
                x_comb_new = np.concatenate((rep, approx_samples), axis=1)
            else:
                x_comb_new = np.concatenate((approx_samples, rep), axis=1)
            all_x_comb = x_comb_new

            mean, var = gp.predict(all_x_comb)
            
            std = np.sqrt(var)

            if use_fixed_kappa:
                ucb_tmp = mean + np.sqrt(kappa) * std
            else:
                ucb_tmp = mean + np.sqrt(kappa_scale * d * np.log(2 * iteration)) * std

            ucb_cum = np.sum(ucb_tmp) / len(approx_samples)

        if USE_DIRECT_OPTIMIZER:
            optimizer_flag = -1
        else:
            optimizer_flag = 1
        
        return optimizer_flag * ucb_cum

    @staticmethod
    def _ucb(x, gp, kappa, use_fixed_kappa, kappa_scale, iteration, gp_model, gp_samples, player_id, x_sim):
        if USE_DIRECT_OPTIMIZER:
            x = x.reshape(1, -1)

        d = x.shape[1]
        
        if player_id == 1:
            x_comb = np.concatenate((x, x_sim.reshape(1, -1)), axis=1)
        elif player_id == 2:
            x_comb = np.concatenate((x_sim.reshape(1, -1), x), axis=1)
        
        mean, var = gp.predict(x_comb)
        std = np.sqrt(var)

        if use_fixed_kappa:
            ucb_tmp = mean + np.sqrt(kappa) * std
        else:
            ucb_tmp = mean + np.sqrt(kappa_scale * d * np.log(2 * iteration)) * std

        if USE_DIRECT_OPTIMIZER:
            optimizer_flag = -1
        else:
            optimizer_flag = 1
        
        return optimizer_flag * ucb_tmp


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog(object):

    def __init__(self, params):

        self.ymax = None
        self.xmax = None
        self.params = params
        self.ite = 1

        self.start_time = datetime.now()
        self.last_round = datetime.now()

        # sizes of parameters name and all
        self.sizes = [max(len(ps), 7) for ps in params]

        # Sorted indexes to access parameters
        self.sorti = sorted(range(len(self.params)),
                            key=self.params.__getitem__)

    def reset_timer(self):
        self.start_time = datetime.now()
        self.last_round = datetime.now()

    def print_header(self, initialization=True):

        if initialization:
            print("{}Initialization{}".format(BColours.RED,
                                              BColours.ENDC))
        else:
            print("{}Bayesian Optimization{}".format(BColours.RED,
                                                     BColours.ENDC))

        print(BColours.BLUE + "-" * (29 + sum([s + 5 for s in self.sizes])) +
            BColours.ENDC)

        print("{0:>{1}}".format("Step", 5), end=" | ")
        print("{0:>{1}}".format("Time", 6), end=" | ")
        print("{0:>{1}}".format("Value", 10), end=" | ")

        for index in self.sorti:
            print("{0:>{1}}".format(self.params[index],
                                    self.sizes[index] + 2),
                  end=" | ")
        print('')

    def print_step(self, x, y, warning=False):

        print("{:>5d}".format(self.ite), end=" | ")

        m, s = divmod((datetime.now() - self.last_round).total_seconds(), 60)
        print("{:>02d}m{:>02d}s".format(int(m), int(s)), end=" | ")

        if self.ymax is None or self.ymax < y:
            self.ymax = y
            self.xmax = x
            print("{0}{2: >10.5f}{1}".format(BColours.MAGENTA,
                                             BColours.ENDC,
                                             y),
                  end=" | ")

            for index in self.sorti:
                print("{0}{2: >{3}.{4}f}{1}".format(
                            BColours.GREEN, BColours.ENDC,
                            x[index],
                            self.sizes[index] + 2,
                            min(self.sizes[index] - 3, 6 - 2)
                        ),
                      end=" | ")
        else:
            print("{: >10.5f}".format(y), end=" | ")
            for index in self.sorti:
                print("{0: >{1}.{2}f}".format(x[index],
                                              self.sizes[index] + 2,
                                              min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")

        if warning:
            print("{}Warning: Test point chose at "
                  "random due to repeated sample.{}".format(BColours.RED,
                                                            BColours.ENDC))

        print()

        self.last_round = datetime.now()
        self.ite += 1

    def print_summary(self):
        pass
