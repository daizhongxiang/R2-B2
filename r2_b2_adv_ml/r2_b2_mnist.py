"""
This script runs the R2-B2 algorithm for adversarial ML using the MNIST dataset
"""

import GPy
import numpy as np
import matplotlib.pyplot as plt
from bayesian_optimization_r2b2 import BayesianOptimization
import pickle

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy

import tensorflow as tf

import os


# latent dimension of the VAE
latent_dim = 2

# level-0 mixed strategy, it supports "random" and "gp_mw"
level_0_pol = "random"
r2b2_lite = False

# The index of the image to attack/defend
attack_ind = 3

# The reasoning levels of Agents 1 and 2
reasoning_level_1 = 0
reasoning_level_2 = 0

# the number of samples used to approximate the expectation in level-1 reasoning
sampling_approximation = 500

# The directory to which the results are saved
log_dir = "results_mnist_" + level_0_pol

# K_gp_mw is the size of the discretized domain (for each agent) that is used by the GP-MW algorithm
# Sicne we assume the domain is 2D for both attacker and defender, when K_gp_mw = 10000 = 100 * 100, each input dimension is discretized into a uniform grid of size 100
K_gp_mw = 10000
domain_file = "sub_domain_K_10000_D_2.pkl"


# The GP hyperparameters are optimized via MLE every gp_opt_schedule iterations
if reasoning_level_1 == 0 and reasoning_level_2 == 0:
    # when reasoning at level 0, we don't need to optimize the GP hyperparameters, and thus set this to a large value
    gp_opt_schedule = 1000
else:
    gp_opt_schedule = 10


func_list = np.arange(0, 10)


# Load the MNIST dataset
num_classes = 10
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_test = x_test.astype('float32')
x_test /= 255
y_test = keras.utils.to_categorical(y_test, num_classes)

# Load the target ML model to be attacked/defended
new_model = keras.models.load_model('mnist_saved_keras_model.h5')
pred = new_model.predict(x_test, verbose=0)

# Load the target image to be attacked/defended
all_img_inds = pickle.load(open("img_inds_mnist.pkl", "rb"))
target_img_ind = all_img_inds[attack_ind]

x = x_test[target_img_ind, :, :, 0]
groundtruth_class = np.argmax(y_test[target_img_ind, :])
print("groundtruth_class: ", groundtruth_class)
x_un_pert = x_test[target_img_ind, :, :, 0] # un-perturbed image


#### Below are the codes to load/initizlize the VAE; some of the parameters/variable here may not be necessary
#### For now, we let the attacker and defender use the same VAE; but the use of two different VAEs can be easily achieved
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

original_dim = 28 * 28

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128

epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)
# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
models = (encoder, decoder)
data = (x_test, y_test)

reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

vae.load_weights("vae_mlp_mnist_D_" + str(latent_dim) + ".h5")


### find the latent encoding (i.e., the mean vector) of the target image
z_mean, _, _ = encoder.predict(x_test[target_img_ind, :, :, 0].reshape(1, -1), batch_size=batch_size)
print("z_mean: ", z_mean.shape)



dim_high = 28

# eps defines the search space for the perturbations with bounded inf norm
eps = 0.2
target_class_ind = 0 # The target class if targeted-attack is used; it is not used in the current versionn, but can be easily adapted

TARGETED_ATTACK = False

def mnist_func_1(param):
    # Payoff function of the attacker
    param = np.array(param)
    param = param * 4 - 2

    param_1 = param[:int(len(param)/2)] # input action of the attacker
    param_2 = param[int(len(param)/2):] # input action of the defender

    # The attacker perturbs the encoded latent vector "z_mean", and then decodes the perturbed vector to an image
    z_pert_1 = z_mean + param_1
    x_decoded_1 = decoder.predict(z_pert_1)
    x_pert_1 = x_decoded_1[0].reshape(dim_high, dim_high)

    # Clip the image to meet the requirement of bounded inf norm
    for i in range(dim_high):
        for j in range(dim_high):
            x_pert_1[i, j] = np.min([x_pert_1[i, j], x_un_pert[i, j] + eps])
            x_pert_1[i, j] = np.max([x_pert_1[i, j], x_un_pert[i, j] - eps])

    # Now the defender encodes the perturbed image by the attacker, adds transformation to the encoded latent vector, and finally decodes the transformed latent vector back to an image, to be passed as input to the target ML model
    z_mean_pert, _, _ = encoder.predict(x_pert_1.reshape(1, -1), batch_size=batch_size)
    z_pert_2 = z_mean_pert + param_2
    x_decoded_2 = decoder.predict(z_pert_2)
    x_pert_2 = x_decoded_2[0].reshape(dim_high, dim_high)

    # Clip the image to meet the requirement of bounded inf norm
    for i in range(dim_high):
        for j in range(dim_high):
            x_pert_2[i, j] = np.min([x_pert_2[i, j], x_pert_1[i, j] + eps])
            x_pert_2[i, j] = np.max([x_pert_2[i, j], x_pert_1[i, j] - eps])

    pred = new_model.predict(x_pert_2.reshape(1, dim_high, dim_high, 1), verbose=0)
    
    if TARGETED_ATTACK:
        if np.argmax(pred[0, :]) == target_class_ind:
            print("[Targeted attack succeeded!]")
        return pred[0, target_class_ind] # targeted attack: return the pred score for the target attack class
    else:
        pred_others = np.append(pred[0, :groundtruth_class], pred[0, groundtruth_class+1:])
        max_score_wrong_class = np.max(pred_others)
        if max_score_wrong_class > pred[0, groundtruth_class]:
            print("[Non-targeted attack succeeded!]")
            print("Altered to class: ", np.argmax(pred[0, :]))

            # return the maximum predictive probability among all incorrect classes, as the payoff for the attacker
            return max_score_wrong_class, True # The second argument is True if untargeted attack is successful, and False otherwise
        else:
            return max_score_wrong_class, False

def mnist_func_2(param):
    # Payoff function of the defender
    param = np.array(param)
    param = param * 4 - 2

    param_1 = param[:int(len(param)/2)] # input action of the attacker
    param_2 = param[int(len(param)/2):] # input action of the defender

    # The attacker perturbs the encoded latent vector "z_mean", and then decodes the perturbed vector to an image
    z_pert_1 = z_mean + param_1
    x_decoded_1 = decoder.predict(z_pert_1)
    x_pert_1 = x_decoded_1[0].reshape(dim_high, dim_high)

    # Clip the image to meet the requirement of bounded inf norm
    for i in range(dim_high):
        for j in range(dim_high):
            x_pert_1[i, j] = np.min([x_pert_1[i, j], x_un_pert[i, j] + eps])
            x_pert_1[i, j] = np.max([x_pert_1[i, j], x_un_pert[i, j] - eps])

    # Now the defender encodes the perturbed image by the attacker, adds transformation to the encoded latent vector, and finally decodes the transformed latent vector back to an image, to be passed as input to the target ML model
    z_mean_pert, _, _ = encoder.predict(x_pert_1.reshape(1, -1), batch_size=batch_size)
    z_pert_2 = z_mean_pert + param_2
    x_decoded_2 = decoder.predict(z_pert_2)
    x_pert_2 = x_decoded_2[0].reshape(dim_high, dim_high)

    # Clip the image to meet the requirement of bounded inf norm
    for i in range(dim_high):
        for j in range(dim_high):
            x_pert_2[i, j] = np.min([x_pert_2[i, j], x_pert_1[i, j] + eps])
            x_pert_2[i, j] = np.max([x_pert_2[i, j], x_pert_1[i, j] - eps])

    pred = new_model.predict(x_pert_2.reshape(1, dim_high, dim_high, 1), verbose=0)

    if TARGETED_ATTACK:
        if np.argmax(pred[0, :]) == target_class_ind:
            print("[Targeted attack succeeded!]")
        return pred[0, target_class_ind] # targeted attack: return the pred score for the target attack class
    else:
        pred_others = np.append(pred[0, :groundtruth_class], pred[0, groundtruth_class+1:])
        max_score_wrong_class = np.max(pred_others)
        if max_score_wrong_class > pred[0, groundtruth_class]:
            # return 1 - (the maximum predictive probability among all incorrect classes), as the payoff for the defender
            return 1 - max_score_wrong_class, False
        else:
            return 1 - max_score_wrong_class, True


for i in func_list:
    if latent_dim == 2:
        if not r2b2_lite:
            log_file_name = log_dir + "/r2b2_mnist_LD_" + str(latent_dim) + "_levels_" \
                    + str(reasoning_level_1) + "_" + \
                    str(reasoning_level_2) + "_approx_samples_" + str(sampling_approximation) + \
                    "_iter_" + str(i) + ".p"
        else:
            log_file_name = log_dir + "/r2b2_mnist_LD_" + str(latent_dim) + "_levels_" \
                    + str(reasoning_level_1) + "_" + \
                    str(reasoning_level_2) + "_approx_samples_" + str(sampling_approximation) + \
                    "_iter_" + str(i) + "_r2b2_lite.p"

        lr_BO = BayesianOptimization(f_1=mnist_func_1, f_2=mnist_func_2,
                pbounds={'x1':(0, 1), 'x2':(0, 1), 'x3':(0, 1), 'x4':(0, 1)}, gp_opt_schedule=gp_opt_schedule, \
                gp_model='gpy', use_init=None, gp_mcmc=False, \
                log_file=log_file_name, save_init=False, \
                save_init_file=None, fix_gp_hypers=None, domain_size=None, \
                level_0_policy_1=level_0_pol, level_0_policy_2=level_0_pol, gp_mw_eta=0.1, \
                reasoning_level_player_1=reasoning_level_1, reasoning_level_player_2=reasoning_level_2, \
                r2b2_light_player_1=r2b2_lite, \
                r2b2_light_player_2=r2b2_lite, ARD=True, sampling_approximation=sampling_approximation, domain_file=domain_file)
        lr_BO.maximize(n_iter=150, init_points=5, kappa=2.0, use_fixed_kappa=False, kappa_scale=0.5, acq='ucb')

