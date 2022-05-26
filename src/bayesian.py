
from IPython.display import clear_output
from math import sqrt
from sklearn.model_selection import train_test_split, KFold
from math import inf
from copy import deepcopy
import time,datetime
import jsonpickle
from sklearn.metrics import  mean_squared_error
from multiprocess import pool
from evaluation import *
from sklearn.gaussian_process import GaussianProcessRegressor
import random as r




def objective(data,models,configs,loss_eval=loss_extractor,preprocess=Preprocess_Std(), n_folds=5,fixed_hyperparams = None,log_file=None):
    extractor_cv_results, predictor_cv_results, _, returned_models = experiment_cv(data, models,
                                                                                   configs,
                                                                                   loss_eval=loss_eval,
                                                                                   log_file=log_file,
                                                                                   fixed_hyperparams=fixed_hyperparams,preprocess=preprocess)

    return extractor_cv_results.score()

def surrogate(fun,configs):
    #todo transform configs into array form
    return [fun.predict(config, return_std=True) for config in configs]


def aquisition(fun,configs,new_configs):
    """
    Probability of a new config improving an old one
    :param fun:
    :param configs:
    :param new_configs:
    :return:
    """
    from scipy.stats import norm

    yhat = surrogate(fun,configs)
    best = max(yhat)

    mu, std = surrogate(fun,new_configs)

    probs = norm.cdf((mu - best) / (std + 1E-9))
    return probs


def opt_acquisition(fun,configs,scores,config_gen):
    """
    Used to sample the next point; ie calculate the most optimal
    :param fun:
    :param scores:
    :param configs:
    :return:
    """
    new_configs = []
    for i in range(0,100):
        new_configs.append(configs.generate_random())

    scores = aquisition(fun,configs,config_gen.to_array(new_configs))

    from numpy import argmax
    ix = argmax(scores)
    return config_gen.from_array(new_configs[ix])


def hyperopt_bo():
    from hyperopt import fmin, tpe, hp
    space = hp.choice('hyperparams',[
        {
            'type':'lr',

        }

    ])


def bayesian_search(data, models, configs,loss_eval=loss_extractor,preprocess=Preprocess_Std(), n_folds=5,log_file = None,n_generations=10,fixed_hyperparams = None ):

    #list all params (refactor configs to hold all)



    X_stack = []
    y_stack = []

    #define our surrogate function -
    surrogate_fun = GaussianProcessRegressor()

    #define our search functionss - random search

    #transform


    #define our aquisition function
    config_gen = RandomConfigGen()
    #run first generation and fit model
    scores = objective(data,models,configs,loss_eval=loss_eval,preprocess=preprocess, n_folds=n_folds,fixed_hyperparams = fixed_hyperparams,log_file=log_file)
    for name in scores.keys():
        X_stack.append(config_gen.to_array(configs[name]))
        y_stack.append(scores[name])
    surrogate_fun.fit(X_stack,y_stack)

    from deep_net import RandomNet


    #search for n generations
    for i in range(n_generations):
        new_config = opt_acquisition(surrogate_fun,X_stack,y_stack,config_gen=config_gen)
        new_model = RandomNet(input_size=n_features,n_layers=new_config.n_layers,act_fun=new_config.act_function) #todo use config params

        y = objective(data,{'gen_{}'.format(i),new_model},{'gen_{}'.format(i),new_config},loss_eval=loss_eval,preprocess=preprocess, n_folds=n_folds,fixed_hyperparams = fixed_hyperparams,log_file=log_file)
        new_config = config_gen.to_array(new_config)
        est = surrogate(surrogate_fun,new_config)
        print('f()=%3f, actual=%.3f' % (est, y))
        #add to data instances
        X_stack.append(new_config)
        y_stack.append(y)
        surrogate_fun.fit(X_stack,y_stack)




