This directory contains the code used in the experiments on adversarial ML (Section 4.2.1) in the paper "R2-B2: Recursive Reasoning-Based Bayesian Optimization for No-Regret Learning in Games".
It includes the implementation of the R2-B2 (as well as R2-B2-Lite) algorithm, and its application in adversarial ML, which corresponds to a two-agent constant-sum game between the attacker and defender.

Dependencies:
* Keras
* Tensorflow
* GPy: https://github.com/SheffieldML/GPy
* scipydirect: needed in order to use the DIRECT algorithm to optimize the acquisition function
    * install scipydirect with "pip install scipydirect", and then replace the content of the script "PYTHON_PATH/lib/python3.5/site-packages/scipydirect/\_\_init\_\_.py" with the content of the script "scipydirect_BO.py"

Description of the scripts:
* bayesian_optimization_r2b2.py: implements the main R2-B2 algorithm
* helper_funcs_r2b2.py: contains some helper functions to support the implementation of the R2-B2 algorithm (bayesian_optimization_r2b2.py)
* r2_b2_mnist.py: runs the R2-B2 algorithm for the MNIST experiment
* r2_b2_cifar_10.py: runs the R2-B2 algorithm for the CIFAR-10 experiment
* generate_discrete_domain_for_gp_mw.py: generates the discretized domain that is required by the GP-MW algorithm, for the MNIST experiment
* generate_target_ML_model_MNIST.py: generates and saves the target ML model to be attacked/defended for MNIST
* generate_target_ML_model_CIFAR.py: generates and saves the target ML model to be attacked/defended for CIFAR-10
* generate_VAE_MNIST.py: generates and saves the VAE to be used by both attacker and defender for dimensionality reduction for MNIST
* generate_VAE_CIFAR.py: generates and saves the VAE to be used by both attacker and defender for dimensionality reduction for CIFAR-10
* analyze_r2_b2_adv_ml.ipynb: is an IPython notebook for analyzing the results from the experiments

Description of the directories:
* results_mnist_random: saves the results for the MNIST experiment, using the random search level-0 strategy
* results_mnist_gp_mw: saves the results for the MNIST experiment, using the GP-MW level-0 strategy
* results_cifar_random: saves the results for the CIFAR-10 experiment, using the random search level-0 strategy
* saved_models: saves the target ML model for CIFAR-10
Only the "results_mnist_random" folder already contains the corresponding results, to avoid excessively large size of the uploaded code.

Description of the files:
* img_inds_mnist.pkl: contains the indices of 1,000 randomly sampled MNIST images that are correctly predicted by the target ML model, from which the images used in the experiments are selected; produced by running "generate_target_ML_model_MNIST.py"
* img_inds_cifar_10.pkl: contains the indices of 1,000 randomly sampled CIFAR-10 images that are correctly predicted by the target ML model, from which the images used in the experiments are selected; produced by running "generate_target_ML_model_CIFAR.py"
* mnist_saved_keras_model.h5: is the saved target ML model for the MNIST experiment
* sub_domain_K_10000_D_2.pkl: contains information about the discretized domain required by the GP-MW level-0 strategy, used for MNIST
* vae_mlp_mnist_D_2.h5: is the saved VAE model for MNIST
* vae_cifar_10_LD_8.h5: is the saved VAE model for CIFAR-10

