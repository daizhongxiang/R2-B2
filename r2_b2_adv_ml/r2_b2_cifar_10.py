"""
This script runs the R2-B2 algorithm for adversarial ML using the CIFAR-10 dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# import cPickle
import GPy
from bayesian_optimization_r2b2 import BayesianOptimization
import pickle

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import cifar10
import keras

import tensorflow as tf

import os

attack_model = 'saved_models/cifar10_ResNet29v2_model.132.h5' # validation accuracy: 0.92320

# latent dimension of the VAE
latent_dim = 8

# eps defines the search space for the perturbations with bounded inf norm
eps = 0.05

attack_img_ind = 403

# we only use the random search level-0 mixed strategy, due to the high dimensionality of the input action
level_0_pol = "random"

r2b2_lite = False
max_iter = 150

sampling_approximation = 1000

log_directory = "results_cifar_" + level_0_pol

reasoning_level_1 = 0
reasoning_level_2 = 0

if reasoning_level_1 == 0 and reasoning_level_2 == 0:
    gp_opt_schedule = 1000
else:
    gp_opt_schedule = 10

func_list = np.arange(0, 10)


num_classes = 10
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Input image dimensions.
input_shape = x_train.shape[1:]
# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

new_model = keras.models.load_model(attack_model)

### load the target img to attack
all_img_inds = pickle.load(open("img_inds_cifar_10.pkl", "rb"))
target_img_ind = all_img_inds[attack_img_ind]

x = x_test[target_img_ind, :, :, 0]
groundtruth_class = np.argmax(y_test[target_img_ind, :])
print("groundtruth class: ", groundtruth_class)
x_un_pert = x_test[target_img_ind, :, :, :] # the mean is subtracted for x_un_pert


################ VAE starts below
img_rows, img_cols, img_chns = 32, 32, 3
intermediate_dim = 128
epsilon_std = 1.0
epochs = 15
filters = 32
num_conv = 3
batch_size = 256

# tensorflow uses channels_last
# theano uses channels_first
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)

# encoder architecture
x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

# mean and variance for latent variables
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

# sampling layer
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])


# decoder architecture
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(int(filters * img_rows / 2 * img_cols / 2), activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, int(img_rows / 2), int(img_cols / 2))
else:
    output_shape = (batch_size, int(img_rows / 2), int(img_cols / 2), filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([x, x_decoded_mean_squash])

# entire model
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
# vae.summary()


# encoder from learned model
encoder = Model(x, z_mean)

# generator / decoder from learned model
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)


vae.load_weights("vae_cifar_10_LD_" + str(latent_dim) +  ".h5")


x_un_pert_add_mean = x_un_pert + x_train_mean
z_mean = encoder.predict(x_un_pert_add_mean.reshape(1, 32, 32, 3))

del x_test, x_train, y_test, y_train

dim_high = 32
target_class_ind = 0 # the target class, if targeted attack is used; it's not used in the current version

TARGETED_ATTACK = False

def cifar_func_1(param):
    ### attacker
    param = np.array(param)
    param = param * 4 - 2

    param_1 = param[:int(len(param)/2)]
    param_2 = param[int(len(param)/2):]
    
    z_pert_1 = z_mean + param_1
    x_decoded_1 = generator.predict(z_pert_1)
    x_pert_1 = x_decoded_1[0].reshape(dim_high, dim_high, 3)
    
    ### clip the image to meet the requirement of bounded inf norm
    for i in range(dim_high):
        for j in range(dim_high):
            for k in range(3):
                x_pert_1[i, j, k] = np.min([x_pert_1[i, j, k], x_un_pert[i, j, k] + eps])
                x_pert_1[i, j, k] = np.max([x_pert_1[i, j, k], x_un_pert[i, j, k] - eps])

    ## re-encode the perturbed image by the Attacker
    z_mean_pert = encoder.predict(x_pert_1.reshape(1, 32, 32, 3), batch_size=batch_size)

    z_pert_2 = z_mean_pert + param_2
    x_decoded_2 = generator.predict(z_pert_2)
    x_pert_2 = x_decoded_2[0].reshape(dim_high, dim_high, 3)
    
    ### clip the image to meet the requirement of bounded inf norm
    for i in range(dim_high):
        for j in range(dim_high):
            for k in range(3):
                x_pert_2[i, j, k] = np.min([x_pert_2[i, j, k], x_pert_1[i, j, k] + eps])
                x_pert_2[i, j, k] = np.max([x_pert_2[i, j, k], x_pert_1[i, j, k] - eps])

    x_pert_2_sub_mean = x_pert_2 - x_train_mean
    pred = new_model.predict(x_pert_2_sub_mean.reshape(1, dim_high, dim_high, 3), verbose=0)

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

            return max_score_wrong_class, True # flag is True if attack is successful
        else:
            return max_score_wrong_class, False

def cifar_func_2(param):
    ### defender
    param = np.array(param)
    param = param * 4 - 2

    param_1 = param[:int(len(param)/2)]
    param_2 = param[int(len(param)/2):]
    
    z_pert_1 = z_mean + param_1
    x_decoded_1 = generator.predict(z_pert_1)
    x_pert_1 = x_decoded_1[0].reshape(dim_high, dim_high, 3)
    
    ### clip the image to meet the requirement of bounded inf norm
    for i in range(dim_high):
        for j in range(dim_high):
            for k in range(3):
                x_pert_1[i, j, k] = np.min([x_pert_1[i, j, k], x_un_pert[i, j, k] + eps])
                x_pert_1[i, j, k] = np.max([x_pert_1[i, j, k], x_un_pert[i, j, k] - eps])

    ## re-encode the perturbed image by the Attacker
    z_mean_pert = encoder.predict(x_pert_1.reshape(1, 32, 32, 3), batch_size=batch_size)

    z_pert_2 = z_mean_pert + param_2
    x_decoded_2 = generator.predict(z_pert_2)
    x_pert_2 = x_decoded_2[0].reshape(dim_high, dim_high, 3)
    
    ### clip the image to meet the requirement of bounded inf norm
    for i in range(dim_high):
        for j in range(dim_high):
            for k in range(3):
                x_pert_2[i, j, k] = np.min([x_pert_2[i, j, k], x_pert_1[i, j, k] + eps])
                x_pert_2[i, j, k] = np.max([x_pert_2[i, j, k], x_pert_1[i, j, k] - eps])

    x_pert_2_sub_mean = x_pert_2 - x_train_mean
    pred = new_model.predict(x_pert_2_sub_mean.reshape(1, dim_high, dim_high, 3), verbose=0)
    
    if TARGETED_ATTACK:
        if np.argmax(pred[0, :]) == target_class_ind:
            print("[Targeted attack succeeded!]")
        return pred[0, target_class_ind] # targeted attack: return the pred score for the target attack class
    else:
        pred_others = np.append(pred[0, :groundtruth_class], pred[0, groundtruth_class+1:])
        max_score_wrong_class = np.max(pred_others)
        if max_score_wrong_class > pred[0, groundtruth_class]:

            return 1 - max_score_wrong_class, False
        else:
            return 1 - max_score_wrong_class, True


for i in func_list:
    if latent_dim == 8:
        if not r2b2_lite:
            log_file_name = log_directory + "/r2b2_cifar_LD_" + str(latent_dim) + \
                        "_levels_" + str(reasoning_level_1) + "_" + \
                        str(reasoning_level_2) + "_approx_samples_" + str(sampling_approximation) + \
                        "_iter_" + str(i) + ".p"
        else:
            log_file_name = log_directory + "/r2b2_cifar_LD_" + str(latent_dim) + \
                        "_levels_" + str(reasoning_level_1) + "_" + \
                        str(reasoning_level_2) + "_approx_samples_" + str(sampling_approximation) + \
                        "_iter_" + str(i) + "_r2b2_lite.p"

        lr_BO = BayesianOptimization(f_1=cifar_func_1, f_2=cifar_func_2,
                pbounds={'x1':(0, 1), 'x2':(0, 1), 'x3':(0, 1), 'x4':(0, 1), \
                        'x5':(0, 1), 'x6':(0, 1), 'x7':(0, 1), 'x8':(0, 1), \
                         'x9':(0, 1), 'x10':(0, 1), 'x11':(0, 1), 'x12':(0, 1), \
                        'x13':(0, 1), 'x14':(0, 1), 'x15':(0, 1), 'x16':(0, 1)}, gp_opt_schedule=gp_opt_schedule, \
                gp_model='gpy', use_init=None, gp_mcmc=False, \
                log_file=log_file_name, save_init=False, \
                save_init_file=None, fix_gp_hypers=None, domain_size=1747, \
                level_0_policy_1=level_0_pol, level_0_policy_2=level_0_pol, gp_mw_eta=0.1, \
                reasoning_level_player_1=reasoning_level_1, reasoning_level_player_2=reasoning_level_2, \
                r2b2_light_player_1=r2b2_lite, \
                r2b2_light_player_2=r2b2_lite, ARD=True, sampling_approximation=sampling_approximation, domain_file=None)
        lr_BO.maximize(n_iter=max_iter, init_points=5, kappa=2.0, use_fixed_kappa=False, kappa_scale=0.5, acq='ucb')

