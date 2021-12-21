import torch
from sklearn.cross_decomposition import PLSCanonical
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from torch import nn
from torch import optim
import torch.functional as F

import numpy as np
from scipy.io import arff
from torch.utils.data import DataLoader, Dataset
import pandas as pd
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
from math import inf


def evo_search(data, models, configs,loss_eval=loss_extractor,preprocess=Preprocess_Std(), n_folds=5,log_file = None,n_generations=10,fixed_hyperparams = None ):
    #todo param to train based on knn
    pop_size = len(models)


    #run first models, save state
    extractor_cv_results, predictor_cv_results, _, returned_models = experiment_cv(data, models, configs,
                                                                                loss_eval=loss_target,
                                                                                log_file=log_file,fixed_hyperparams=fixed_hyperparams)
    prev_models = {'init'+name: model for name, model in models.items()}
    prev_scores = {'init'+name: score for name, score in predictor_cv_results.score().items()}  # todo fix this
    prev_configs = {'init'+name: config for name, config in configs.items()}

    #todo track best model and score

    current_models = {'gen_0_'+key: deepcopy(value) for key, value in models.items()}
    current_configs= {'gen_0_'+key: deepcopy(value) for key, value in configs.items()}

    evo_results_loss = pd.DataFrame({'gen': [], 'model': [], 'score': []})
    evo_results_predict = pd.DataFrame({'gen': [], 'model': [], 'score': []})

    #calc best
    best_model= None
    best_score = inf


    k,v = min(prev_scores.items(),key = lambda x : x[1])
    best_score = v
    best_model = deepcopy(prev_models[k])
    init_best_score = best_score

    # append initial results to loss
    for name, score in extractor_cv_results.score().items():
        evo_results_predict = evo_results_predict.append({'gen': -1, 'model': name, 'score': score}, ignore_index=True)
    for name, score in predictor_cv_results.score().items():
        evo_results_loss = evo_results_loss.append({'gen': -1, 'model': name, 'score': score}, ignore_index=True)

    #run a first generation
    for gen in range(n_generations):
        utils.log("Running generation {}".format(gen))
        #mutating
        current_models = {key: value.mutate() for key, value in current_models.items()}
        #run generation
        extractor_cv_results, predictor_cv_results,_, returned_models = experiment_cv(data, current_models, current_configs, loss_eval=loss_target,
                                                                 log_file=log_file, fixed_hyperparams=fixed_hyperparams)
        scores = predictor_cv_results.score()

        #calculate bests
        k, v = min(scores.items(), key=lambda x: x[1])
        if v < best_score:
            best_score = v
            best_model = deepcopy(current_models[k])

        #append to results
        for name,score in scores.items():
            evo_results_predict = evo_results_predict.append({'gen':gen,'model':name,'score':score}, ignore_index=True)
        scores = predictor_cv_results.score()
        for name, score in scores.items():
            evo_results_loss = evo_results_loss.append({'gen': gen, 'model': name, 'score': score}, ignore_index=True)

        # todo track best model and score

        #select from models, prev gen
        from numpy.random import choice
        total_score = sum([1/val for name, val in prev_scores.items()]) + sum([1/val for name, val in scores.items()])

        dk = []
        dp = []

        for k,p in prev_scores.items():
            dk.append(k)
            dp.append(1.0/p/total_score)

        for k,p in scores.items():
            dk.append(k)
            dp.append(1.0/p/total_score)


        draw = choice(dk,pop_size,p=dp,replace=False)
        print(len(draw))
        assert len(draw) == pop_size

        #assign_new models
        new_models = {}
        new_configs = {}
        new_scores = {}

        #select our models
        for key in draw:
            if key in new_models.keys():
                print("with replacement_{}".format(key))
            if key in prev_models.keys():
                if key in current_models.keys():
                    print("ayye_{}".format(key))
                new_models[key]= prev_models[key]
                new_configs[key] = prev_configs[key]
                new_scores[key] = prev_scores[key]
            elif key in current_models.keys():
                new_models[key] = current_models[key]
                new_configs[key] = current_configs[key]
                new_scores[key] = scores[key]
            else:
                print("key error after draw: ".format(key))

        assert len(new_models) == pop_size

        #now we save the current models and a copy that will be mutated


        #leave current models as previous models
        prev_models = {name:deepcopy(value) for name, value in new_models.items()}
        prev_configs = {name:deepcopy(value) for name, value in new_configs.items()}
        prev_scores = {name:deepcopy(value) for name, value in new_scores.items()}

        assert len(prev_models) == pop_size

        #mutate pop and assign to current models
        current_models = {'gen_{}_'.format(gen+1)+key:deepcopy(value) for key,value in new_models.items()}
        current_configs = {'gen_{}_'.format(gen+1)+key:deepcopy(value) for key,value in new_configs.items()}

        assert len(current_models) == pop_size


        utils.log("Finished generation_{}".format(gen))

    #todo track best model
    utils.log("Finished Evo Search")
    utils.log("Improved from {} to {}".format(init_best_score,best_score))

    return evo_results_loss, evo_results_predict
