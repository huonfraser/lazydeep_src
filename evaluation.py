from abc import ABC
from ctypes import Union
from pprint import pprint

import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from torch import nn
from torch import optim
import torch.functional as F
from torch.utils.data import DataLoader, Dataset,Subset
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression,PLSCanonical
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

import argparse
import time,datetime
import logging
import jsonpickle
from copy import deepcopy
from math import inf
from math import sqrt

from sk_models import LocalWeightedRegression
from splitters import *
from utils import *
from lazydeep import *
import math
from math import log10, inf

def loss_encoder(X, y, model_output,loss_fun):
    """
    Function to calculate loss for an autoencoder type networks
    :param X:
    :param y:
    :param model_output:
    :param loss_fun:
    :return:
    """
    return loss_fun(X,model_output)

def loss_target(X, y, model_output,loss_fun):
    """
    Function to calculate loss based of targets
    :param X:
    :param y: the target matrix
    :param model_output:
    :param loss_fun:
    :return:
    """
    return loss_fun(y,model_output)

class ModelScheme():

    def __init__(self,time=False):
        self.time=time

    def pretrain(self,model,data:TabularDataset):
        return None, None

    def train(self,model, data:TabularDataset,val_data=None):
        """
        Take in models, dataset, (optional validation set),
        :param model:
        :param data:
        :param val_data:
        :return: trained version of the input models, training score
        """
        return None, None

    def test(self,model, data:TabularDataset,test_type="test"):
        """

        :param model:
        :param data:
        :param test_type:
        :return: test scores, and predictions
        """
        return None, None

class DeepAndLWR(ModelScheme):
    def __init__(self, configs, fixed_hyperparams=None,
                 logger="",
                 loss_eval=loss_target,
                 tensorboard=None,
                 device= "cpu",
                 adaptive_lr=False,
                 n_neighbours = 100,
                 *args,
                 **kwargs):
        super(DeepAndLWR,self).__init__(*args,**kwargs)
        self.lwr_models = None
        self.configs = configs
        self.fixed_hyperparams = fixed_hyperparams
        self.logger = logging.getLogger(logger)
        self.loss_eval = loss_eval
        self.loss_fun = mean_squared_error
        self.loss_fun_torch = nn.MSELoss()
        self.tensorboard = tensorboard
        self.device = device
        self.adaptive_lr = adaptive_lr

        self.n_neighbours = n_neighbours

        #setup our current schemes
        self.deep_scheme = DeepScheme(
            configs = configs,
            fixed_hyperparams = fixed_hyperparams,
            loss_eval = loss_eval,
            tensorboard = tensorboard,
            device = device,
            adaptive_lr = adaptive_lr,
            logger = logger)
        self.lwr_scheme = DeepLWRScheme_n_to_1(n_neighbours=n_neighbours,logger=logger)


    def pretrain(self,models, train_data):
        self.deep_scheme.pretrain(models,train_data)
        self.lwr_scheme.pretrain(models,train_data)

    def train(self,models, train_data, val_data=None):
        trained_models_deep,train_scores_deep = self.deep_scheme.train(models,train_data,val_data)

        trained_models_lwr,train_scores_lwr = self.lwr_scheme.train(trained_models_deep,train_data,val_data)

        train_scores_1 = {k:v for k,v in train_scores_deep.items()}
        train_scores_2 = {k + "_lwr": v for k, v in train_scores_lwr.items()}

        #train_models_1 = {k+"_deep":v for k,v in trained_models_deep.items()}
        #train_models_2 = {k + "_lwr": v for k, v in trained_models_lwr.items()}
        #trained_models = {**train_models_1, **train_models_2}
        train_scores = {**train_scores_1, **train_scores_2}
        return trained_models_lwr, train_scores

    def test(self,models, test_data, test_type="test"):
        # train deep
        test_scores_deep, preds_deep = self.deep_scheme.test(models,test_data, test_type)
        # train lwr
        test_scores_lwr, preds_lwr = self.lwr_scheme.test(models,test_data, test_type)

        test_scores_1 = {k:v for k,v in test_scores_deep.items()}
        test_scores_2 = {k + "_lwr": v for k, v in test_scores_lwr.items()}

        preds_1= {k:v for k,v in preds_deep.items()}
        preds_2 = {k + "_lwr": v for k, v in preds_lwr.items()}

        preds = {**preds_1, **preds_2}
        #del preds['y_lwr']

        loss_scores = {**test_scores_1, **test_scores_2}
        #pprint(loss_scores)
        return loss_scores,preds



class DeepScheme(ModelScheme):

    def __init__(self, configs, fixed_hyperparams=None,
                 logger="",
                 loss_eval=loss_target,
                 tensorboard=None,
                 device= "cpu",
                 adaptive_lr=False,
                 update = True,
                 *args,
                 **kwargs):
        super(DeepScheme,self).__init__(*args,**kwargs)
        self.configs = configs
        self.fixed_hyperparams = fixed_hyperparams
        self.logger = logging.getLogger(logger)
        self.loss_eval = loss_eval
        self.loss_fun = mean_squared_error
        self.loss_fun_torch = nn.MSELoss()
        self.tensorboard = tensorboard
        self.device = device
        self.update = update
        self.adaptive_lr = adaptive_lr

    def pretrain(self,models,train_data):
        train_X,train_y = zip(*[(X,y) for X, y in train_data])
        for name,model in models.items():
            model.prefit(train_X,train_y)


    def train(self,models,train_data,val_data=None):
        if self.update:
            """
            Train a pytorch type model
            """
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            train_start = datetime.datetime.now()

            bs = self.fixed_hyperparams['bs']
            loss = self.fixed_hyperparams['loss']
            epochs = self.fixed_hyperparams['epochs']

            train_loader = DataLoader(train_data, batch_size=bs, shuffle=False)

            # setup optimisers and schedulers
            opts = {}
            schedulers = {}
            for i, (name, model) in enumerate(models.items()):
                config = self.configs[name]
                opt = config.opt(model.parameters(), lr=config.lr)
                opts[name] = opt

                if config.lr_update is None:
                    schedulers[name] = None
                elif config.lr_update == optim.lr_scheduler.ExponentialLR:
                    scheduler = optim.lr_scheduler.ExponentialLR(opt, 0.95)
                    schedulers[name] = scheduler
                elif config.lr_update == optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, verbose=True)
                    schedulers[name] = scheduler
                elif config.lr_update == optim.lr_scheduler.CosineAnnealingLR:
                    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
                    schedulers[name] = scheduler
                elif config.lr_update == optim.lr_scheduler.CosineAnnealingWarmRestarts:
                    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 20)
                    schedulers[name] = scheduler
                else:
                    print(config.lr_update.__name__)

            # record best model and state
            best_losses = {name: inf for name in models.keys()}
            best_models = {name: deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}) for name, model in
                           models.items()}
            best_lrs = {name: inf for name in models.keys()}

            self.logger.info("Training extractors on {} instances, validating on {} instances, for {} epochs".format(
                len(train_data), len(val_data), epochs))

            n_train_batches = 0

            for epoch in range(0, epochs):
                self.logger.info("\n--- EPOCH {}---".format(epoch))
                epoch_start = datetime.datetime.now()

                # train
                epoch_train_total = {name: 0 for name in models.keys()}
                n_observed = 0
                for batch, (X, y) in enumerate(train_loader):
                    # if self.preprocess is not None:
                    #    X = X.detach().cpu().float()
                    #    X = torch.Tensor(self.preprocess.transform(X)).to(device).float()
                    # else:

                    X = X.to(self.device).float()
                    y = y.to(self.device).float()

                    bs = y.shape[0]
                    n_observed += bs

                    batch_losses = train_batch(X, y, models, opts, self.loss_eval, self.loss_fun_torch)
                    epoch_train_total = {name: epoch_train_total[name] + np.sum(batch_losses[name] * bs) for name in
                                         models.keys()}
                    # if not self.tensorboard is None:
                    #    self.tensorboard.add_scalars("batch/train/loss", batch_losses, n_train_batches)
                    n_train_batches += 1
                    del X, y
                    torch.cuda.empty_cache()


                from math import log10
                # log training
                epoch_train_scores = {name: value / n_observed for name, value in epoch_train_total.items()}
                self.logger.info("Extractor Train Losses are " + ",".join(
                    str(i) + ":" + str(round(j, 4)) + "(" + str(round(log10(opts[i].param_groups[0]['lr']), 6)) + ")" for
                    (i, j) in epoch_train_scores.items()))
                if not self.tensorboard is None:
                    self.tensorboard.add_scalars("epoch/train/loss", epoch_train_scores, epoch)

                epoch_middle = datetime.datetime.now()

                # validate
                epoch_val_scores, epoch_val_preds = self.test(models,val_data,test_type="val")
                if not self.tensorboard is None:
                    self.tensorboard.add_scalars("epoch/val/loss", epoch_train_scores, epoch)

                # run lr schedulers. Either restart models or run schedulers
                if self.adaptive_lr:
                    iterate_schedulers(schedulers=schedulers, opts=opts, models=models, scores=epoch_train_scores)

                # store current best model
                for name, model in models.items():
                    if epoch_val_scores[name] < best_losses[name]:
                        best_losses[name] = epoch_val_scores[name]
                        best_models[name] = deepcopy(model.state_dict()) #deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
                        best_lrs[name]= opts[name].param_groups[0]['lr']

                epoch_end = datetime.datetime.now()
                epoch_diff = (epoch_end - epoch_start)

                del epoch_val_scores, epoch_val_preds, epoch_train_total, batch_losses
                self.logger.info("Epoch {} finished in {} ".format(epoch, epoch_diff))
            self.logger.info("\n-----------")
            self.logger.info("Finished training extractors with a best validation loss of " + ",".join(
                str(i) + ":" + str(round(j, 4)) for (i, j) in best_losses.items()))

            # return best models along with validation results

            train_end = datetime.datetime.now()
            train_diff = train_end - train_start
            self.logger.info("Training took {}".format(train_diff))
            # final learning rates
            #final_learning_rates = {name:opts[name].param_groups[0]['lr'] for name in self.models.keys()}
            #for name, config in self.configs.items():
            #    config.lr = final_learning_rates[name
            #   ]

            #set model states to the best for returning
            for name,model in models.items():
                model.load_state_dict(best_models[name])


            return models, best_losses
        else:
            return models, None

    def test(self,models,data,test_type="test"):
        """'=
        Test a pytorch type model
        Log each batch to pytorch
        """
        bs = self.fixed_hyperparams['bs']
        loss = self.fixed_hyperparams['loss']
        #todo process fit
        #if self.preprocess is not None:
        #    data = self.preprocess.transform(data)

        test_start = datetime.datetime.now()
        dl = DataLoader(data, bs, shuffle=False, num_workers=0)

        # test_diffs = {}
        preds = {name: [] for name in models.keys()}
        preds['y'] = []

        # utils.log("Testing on {} instances".format(len(dl.dataset)),log_file)
        with torch.no_grad():
            test_running_total = {name: 0 for name in models.keys()}
            n_observed = 0
            for batch, (X, y) in enumerate(dl):
                #if self.preprocess is not None:
                #     X = X.detach().cpu().float()
                #    X = torch.Tensor(self.preprocess.transform(X)).to(device).float()
                #else:
                X = X.to(self.device).float()
                y = y.to(self.device).float()
                bs = y.shape[0]
                n_observed += bs

                batch_preds, batch_losses = test_batch(X, y, models, None, self.loss_eval, self.loss_fun_torch)
                batch_preds['y'] = y.detach().cpu().tolist()
                test_running_total = {name: test_running_total[name] + np.sum(batch_losses[name]) * bs for name in
                                      models.keys()}
                #if not self.tensorboard is None:
                #    self.tensorboard.add_scalars(f"batch/{test_type}/loss", batch_losses, batch)
                test_scores = {name: value / n_observed for name, value in test_running_total.items()}
                preds = {name: preds[name]+batch_preds[name] for name in preds.keys()}
        self.logger.info("Tested ({}) on {} instances with mean losses of: ".format(test_type,n_observed) + ",".join(
            [str(i) + ":" + str(round(j, 4)) for i, j in test_scores.items()]))

        test_end = datetime.datetime.now()
        test_diff = test_end - test_start
        self.logger.info("Testing ({}) took {}".format(test_type,test_diff))

        return test_scores,preds

class SKLearnScheme(ModelScheme):

    def __init__(self,loss_fun_sk = mean_squared_error,logger="",*args,**kwargs):
        super(SKLearnScheme,self).__init__(*args,**kwargs)
        self.loss_fun_sk = loss_fun_sk
        self.logger = logging.getLogger(logger)

    def train(self,models,train_data,val_data=None):
        train_X,train_y = zip(*[(X,y) for X, y in train_data])
        train_X = np.asarray(train_X)
        train_y = np.asarray(train_y)
        nrow, ncol = train_X.shape
        n_features = ncol
        #todo should be name
        fold_scores = {}
        for name,model in models.items():
            model.fit(train_X, train_y)
            y_pred = model.predict(train_X)
            score = self.loss_fun_sk(train_y, y_pred)
            fold_scores[name] = score
        self.logger.info("Finished training SKLearn with a train loss of " + ",".join(
            str(i) + ":" + str(round(j, 4)) for (i, j) in fold_scores.items()))
        return models,fold_scores

    def test(self,models, test_data, test_type="test"):
        test_X,test_y = zip(*[(X,y) for X, y in test_data])
        test_X = np.asarray(test_X)
        test_y = np.asarray(test_y)
        preds = {name: [] for name in models.keys()}
        preds['y'] = test_y.tolist()

        fold_scores = {}
        for name, model in models.items():

            # test model and score
            y_pred = np.squeeze(model.predict(test_X))
            score = self.loss_fun_sk(test_y, y_pred)

            fold_scores[name] = score
            preds[name] = preds[name] + y_pred.tolist()
        self.logger.info("Tested ({}) on {} instances with mean losses of: ".format(test_type,len(test_X)) + ",".join(
            [str(i) + ":" + str(round(j, 4)) for i, j in fold_scores.items()]))
        return fold_scores, preds

class DeepLWRScheme_1_to_n(ModelScheme):
    """
    Take pretrained deep models and run a bunch of locally weighted regressions
    """
    def __init__(self,lwr_models = None,n_neighbours=500,loss_fun_sk = mean_squared_error,logger="",*args,**kwargs):
        super(DeepLWRScheme_1_to_n,self).__init__(*args,**kwargs)
        self.loss_fun_sk = loss_fun_sk
        self.n_neighbours=n_neighbours
        self.lwr_models= lwr_models # {name:LocalWeightedRegression(n_neighbours=self.n_neighbours) for name in models.keys()}
        self.logger = logging.getLogger(logger)
        self.logger_name = logger

    def pretrain(self,models,train_data):
        pass

    def train(self,models,train_data,val_data=None):
        if self.lwr_models is None:
            self.lwr_models= {name: LocalWeightedRegression(n_neighbours=self.n_neighbours) for name in models.keys()}
        fold_scores = {}
        train_X = np.asarray([ X for X, y in train_data])
        train_y = np.asarray([ y for X, y in train_data])
        for i,deep_name in enumerate(models.keys()):
            if i == 0: #if more than one model take the first
                X_t = models[deep_name].compress(torch.tensor(train_X).float()).detach().numpy()

                for lwr_name,lwr_model in self.lwr_models.items():
                    self.lwr_models[lwr_name].fit(X_t,train_y)
                    y_pred = self.lwr_models[lwr_name].predict(X_t)
                    score = self.loss_fun_sk(train_y, y_pred)
                    fold_scores[lwr_name] = score
        self.logger.info("Finished training DeepLWR with a train loss of " + ",".join(           str(i) + ":" + str(round(j, 4)) for (i, j) in fold_scores.items()))
        return models,fold_scores,

    def test(self,models,test_data, test_type="test"):
        test_X,test_y = zip(*[(X,y) for X, y in test_data])
        test_X = np.asarray(test_X)
        test_y = np.asarray(test_y)
        preds = {name: [] for name in self.lwr_models.keys()}
        preds['y'] = test_y.tolist()

        fold_scores = {}
        for i,(deep_name, model) in enumerate(models.items()):
            if i == 0:
                test_X_t = model.compress(torch.tensor(test_X).float()).detach().numpy()
                for lwr_name,lwr_model in self.lwr_models.items():
                # test model and score

                    y_pred = self.lwr_models[lwr_name].predict(test_X_t)
                    score = self.loss_fun_sk(test_y, y_pred)

                    fold_scores[lwr_name] = score
                    preds[lwr_name] = preds[lwr_name] + y_pred.tolist()

        self.logger.info("Tested ({}) on {} instances with mean losses of: ".format(test_type,len(test_X)) + ",".join(
            [str(i) + ":" + str(round(j, 4)) for i, j in fold_scores.items()]))
        return fold_scores, preds


class DeepLWRScheme_n_to_1(ModelScheme):
    """
    Take pretrained deep models and run a bunch of locally weighted regressions
    """
    def __init__(self,lwr_models = None,n_neighbours=500,loss_fun_sk = mean_squared_error,logger="",*args,**kwargs):
        super(DeepLWRScheme_n_to_1,self).__init__(*args,**kwargs)
        self.loss_fun_sk = loss_fun_sk
        self.n_neighbours=n_neighbours
        self.lwr_models= lwr_models # {name:LocalWeightedRegression(n_neighbours=self.n_neighbours) for name in models.keys()}
        self.logger = logging.getLogger(logger)
        self.logger_name = logger

    def pretrain(self,models,train_data):
        pass

    def train(self,models,train_data,val_data=None):
        if self.lwr_models is None:
            self.lwr_models= {name: LocalWeightedRegression(n_neighbours=self.n_neighbours) for name in models.keys()}
        fold_scores = {}
        train_X = np.asarray([ X for X, y in train_data])
        train_y = np.asarray([ y for X, y in train_data])
        for name in models.keys():
            X_t = models[name].compress(torch.tensor(train_X).float()).detach().numpy()
            self.lwr_models[name].fit(X_t,torch.tensor(train_y))
        #pass through our models and build our database
            y_pred = self.lwr_models[name].predict(X_t)
            score = self.loss_fun_sk(train_y, y_pred)
            fold_scores[name] = score
        self.logger.info("Finished training DeepLWR with a train loss of " + ",".join(           str(i) + ":" + str(round(j, 4)) for (i, j) in fold_scores.items()))
        return models,fold_scores,

    def test(self,models,test_data, test_type="test"):
        test_X,test_y = zip(*[(X,y) for X, y in test_data])
        test_X = np.asarray(test_X)
        test_y = np.asarray(test_y)
        preds = {name: [] for name in models.keys()}
        preds['y'] = test_y.tolist()

        fold_scores = {}
        for name, model in models.items():
            test_X_t = model.compress(torch.tensor(test_X).float()).detach().numpy()
            # test model and score
            y_pred = self.lwr_models[name].predict(test_X_t)
            score = self.loss_fun_sk(test_y, y_pred)

            fold_scores[name] = score
            preds[name] = preds[name] + y_pred.tolist()

        self.logger.info("Tested ({}) on {} instances with mean losses of: ".format(test_type,len(test_X)) + ",".join(
            [str(i) + ":" + str(round(j, 4)) for i, j in fold_scores.items()]))
        return fold_scores, preds

class PCAScheme(ModelScheme):
    """
    Scheme to take a bunch of sklearn models, and preprocess with a PLS
    either uses an identical n_components (n_components is int)
    or uses a dict that matches by key
    """
    def __init__(self,whiten=False, n_components=5, loss_fun_sk = mean_squared_error, logger="", *args, **kwargs):
        super(PCAScheme,self).__init__(*args,**kwargs)
        self.whiten=whiten
        self.n_components = n_components
        self.pca_models = None
        self.loss_fun_sk = loss_fun_sk
        self.logger = logging.getLogger(logger)

        #we can give a static n_components in which case applies to all
        #else use a dict

    def train(self,models,train_data,val_data=None):
        self.pca_models = {}
        # fit pls

        train_X_g = (X for X, y in train_data)
        train_y_g = (y for X, y in train_data)
        train_X = np.asarray([X for X in train_X_g])
        train_y = np.asarray([y for y in train_y_g])
        nrow, ncol = train_X.shape
        n_features = ncol

        #if self.n_components is None:
        #    self.n_components = n_features

        #todo should be name
        fold_scores = {}
        for name,model in models.items():
            if isinstance(self.n_components,int):
                pca = PCA(n_components=self.n_components, whiten=self.whiten)
            else:
                pca = PCA(n_components=self.n_components[name], whiten = self.whiten)
            pca.fit(train_X)
            self.pca_models[name] = pca

            X_train_c = pca.transform(train_X)
            model.fit(X_train_c, train_y)
            y_pred = model.predict(X_train_c)
            score = self.loss_fun_sk(train_y, y_pred)
            fold_scores[name] = score
        self.logger.info("Finished training PLS with a train loss of " + ",".join(
            str(i) + ":" + str(round(j, 4)) for (i, j) in fold_scores.items()))
        return models,fold_scores

    def test(self,models,test_data,test_type="test"):
        """FMang
        Internal method to apply pls feature extraction before a bunch of models

        :param models:
        :param train_data:
        :param test_data:
        :param n_components:
        :param scale:
        :return: a dict of scores and one of predictiosn
        """
        #todo replace with zip
        test_X_g = (X for X, y in test_data)
        test_y_g = (y for X, y in test_data)

        test_X = np.asarray([X for X in test_X_g])
        test_y = np.asarray([y for y in test_y_g])

        preds = {name: [] for name in models.keys()}
        preds['y'] = test_y.tolist()

        fold_scores = {}
        for name, model in models.items():
            # transform data
            X_test_c= self.pca_models[name].transform(test_X)
            # test model and score
            y_pred = model.predict(X_test_c)
            score = self.loss_fun_sk(test_y, y_pred)

            fold_scores[name] = score
            preds[name] = preds[name] + y_pred.tolist()

        self.logger.info("Tested ({}) on {} instances with mean losses of: ".format(test_type,len(test_X)) + ",".join(
            [str(i) + ":" + str(round(j, 4)) for i, j in fold_scores.items()]))

        return fold_scores, preds



class PLSScheme(ModelScheme):
    """
    Scheme to take a bunch of sklearn models, and preprocess with a PLS
    either uses an identical n_components (n_components is int)
    or uses a dict that matches by key
    """
    def __init__(self,scale=True,n_components=5,loss_fun_sk = mean_squared_error,logger="",*args,**kwargs):
        super(PLSScheme,self).__init__(*args,**kwargs)
        self.scale=scale
        self.n_components = n_components
        self.pls_models = None
        self.loss_fun_sk = loss_fun_sk
        self.logger = logging.getLogger(logger)

        #we can give a static n_components in which case applies to all
        #else use a dict

    def train(self,models,train_data,val_data=None):
        self.pls_models = {}
        # fit pls

        train_X_g = (X for X, y in train_data)
        train_y_g = (y for X, y in train_data)
        train_X = np.asarray([X for X in train_X_g])
        train_y = np.asarray([y for y in train_y_g])
        nrow, ncol = train_X.shape
        n_features = ncol

        #if self.n_components is None:
        #    self.n_components = n_features

        #todo should be name
        fold_scores = {}
        for name,model in models.items():
            if isinstance(self.n_components,int):
                pls = PLSRegression(n_components=self.n_components, scale=self.scale)
            else:
                pls = PLSRegression(n_components=self.n_components[name], scale=self.scale)
            pls.fit(train_X, train_y)
            self.pls_models[name] = pls

            X_train_c, y_train_c = pls.transform(train_X, train_y)
            model.fit(X_train_c, train_y)
            y_pred = model.predict(X_train_c)
            score = self.loss_fun_sk(train_y, y_pred)
            fold_scores[name] = score
        self.logger.info("Finished training PLS with a train loss of " + ",".join(
            str(i) + ":" + str(round(j, 4)) for (i, j) in fold_scores.items()))
        return models,fold_scores

    def test(self,models,test_data,test_type="test"):
        """
        Internal method to apply pls feature extraction before a bunch of models

        :param models:
        :param train_data:
        :param test_data:
        :param n_components:
        :param scale:
        :return: a dict of scores and one of predictiosn
        """
        #todo replace with zip
        test_X_g = (X for X, y in test_data)
        test_y_g = (y for X, y in test_data)

        test_X = np.asarray([X for X in test_X_g])
        test_y = np.asarray([y for y in test_y_g])

        preds = {name: [] for name in models.keys()}
        preds['y'] = test_y.tolist()

        fold_scores = {}
        #pprint(models)
        for name, model in models.items():
            # transform data
            X_test_c, y_test_c = self.pls_models[name].transform(test_X, test_y)
            #print(f"{name} - {X_test_c.shape} -{y_test_c.shape}")
            # test model and score
            y_pred = model.predict(X_test_c)
            score = self.loss_fun_sk(test_y, y_pred)

            fold_scores[name] = score
            preds[name] = preds[name] + y_pred.tolist()

        self.logger.info("Tested ({}) on {} instances with mean losses of: ".format(test_type,len(test_X)) + ",".join(
            [str(i) + ":" + str(round(j, 4)) for i, j in fold_scores.items()]))

        return fold_scores, preds



class SplitScheme():
    def __init__(self, random_state = None, preprocessing = None, tensorboard = None,time=False):
        self.preprocessing = preprocessing
        self.tensorboard = tensorboard
        self.random_state = random_state

        self.time = time

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def build(self):
        pass

class CrossValEvaluation(SplitScheme):
    def __init__(self,n_folds=5,*args,**kwargs):
        super(CrossValEvaluation,self).__init__(*args,**kwargs)
        self.n_folds = n_folds
        self.tt_splitter= train_test_split
        self.cv_splitter = K2Fold(n_splits=self.n_folds,random_state=self.random_state)

    #todo resets

    def evaluate(self, original_models, data:TabularDataset, eval:ModelScheme,pretrain=False,logger_name="",load_fun = None):
        logger = logging.getLogger(logger_name)
        logger.info("Running Cross Evaluation with {} folds".format(self.n_folds))

        # initialise datastructures for results
        model_states = {}
        preds = None
        total_scores = None
        scores = None
        #todo preds['y'] = []
        #todo preds['set_id'] = []
        

        model_states['init_state'] = {k:deepcopy(v.state()) for k,v in original_models.items()}

        train_times = {}
        test_times = {}

        train_split,test_split = self.tt_splitter([i for i in range(0,len(data))],train_size=5/6,shuffle=False)
        for fold, (train_ind, val_ind, test_ind) in enumerate(self.cv_splitter.split(train_split)):
            
            if not self.preprocessing is None:
                train_X,train_y = data[train_ind]
                self.preprocessing.fit(train_X,np.squeeze(train_y))
            
            train_data, val_data, test_data = data.split(train_ind, val_ind, test_ind, preprocessing=self.preprocessing)

            logger.info(f"-----------------------------------"
                             f"Fold {fold} - Train {len(train_data)} - Val {len(val_data)} - Test {len(test_data)}"
                             f"-----------------------------------")

            start_time = datetime.datetime.now()
            if load_fun is None:
                models = deepcopy(original_models)
            else:
                models = {name:load_fun(name,model,fold) for name, model in original_models.items()}
 
                
                                        #pretrain
            if pretrain:
                #start = datetime.datetime.now()
                logger.info("Pretraining our models")
                eval.pretrain(models,train_data)
                #end = datetime.datetime.now()
                #self.logger.info(f"Pretraining took {end - start}")
            # train
            trained_models_fold, train_scores_fold = eval.train(models,train_data, val_data)
            end_time = datetime.datetime.now()
            train_times[f"fold_{fold}"] = (end_time-start_time).seconds
            model_states[f"fold_{fold}"] = {k:deepcopy(v.state()) for k,v in trained_models_fold.items()}

            
            #test
            start_time = datetime.datetime.now()
            test_scores_fold, preds_fold = eval.test(trained_models_fold,test_data)
            end_time = datetime.datetime.now()
            test_times[f"fold_{fold}"] = (end_time-start_time).seconds

            #record our predictions
            if preds is None:
                preds = {name:[] for name in preds_fold.keys()}
                preds['set_id'] = []
            for name in preds_fold.keys():
                preds[name] = preds[name] + preds_fold[name]
            preds['set_id'] = preds['set_id'] + [fold for i in val_ind]

            # record our scores
            if scores is None:
                scores = {}
                total_scores = {name:0 for name in test_scores_fold.keys()}
            scores[f"fold_{fold}"] = test_scores_fold
            total_scores = {name: total_scores[name] + test_scores_fold[name] for name in total_scores.keys()}
        scores["MSE"] = {key: mean_squared_error(preds["y"],preds[key]) for key in total_scores.keys()}
        scores["R2"] = {key: r2_score(preds["y"],preds[key]) for key in total_scores.keys()}
        #pprint(scores)

        preds = pd.DataFrame(preds)

        if self.time:
            train_times['mean'] = np.mean([i for i in train_times.values()])
            test_times['mean'] = np.mean([i for i in test_times.values()])
            return scores, preds, model_states, train_times,test_times
        else:
            return scores, preds, model_states

    def build(self, original_models, data: TabularDataset, eval: ModelScheme, pretrain=False, logger_name="",load_fun = None):
        logger = logging.getLogger(logger_name)
        train_times = {}
        test_times = {}

        train_split1,test_split = self.tt_splitter([i for i in range(0,len(data))],train_size=5/6,random_state=self.random_state,shuffle=False)
        train_split,val_split = self.tt_splitter(train_split1,train_size=4/5,random_state=self.random_state,shuffle=False)
        
        if not self.preprocessing is None:
            train_X,train_y = data[train_split]
            print(train_X)
            print(train_y)
            self.preprocessing.fit(train_X,np.squeeze(train_y))
            
        train_data, val_data, test_data = data.split(train_split, val_split, test_split,preprocessing=self.preprocessing)
        logger.info(f"Building final model - Train {len(train_data)} - Test {len(test_data)}")

        start_time = datetime.datetime.now()
        if load_fun is None:
            models = deepcopy(original_models)
        else:
            models = {name: load_fun(name,model) for name,model in original_models.items()}
        
        
        if pretrain:
        #start = datetime.datetime.now()
            logger.info("Pretraining our models")
            eval.pretrain(models,train_data)
            #end = datetime.datetime.now()
            #self.logger.info(f"Pretraining took {end-start}")
        trained_models, train_scores = eval.train(models,train_data,val_data)
        #train
        end_time = datetime.datetime.now()
        train_times[f"train"] = (end_time - start_time).seconds
        #test
        start_time = datetime.datetime.now()
        test_scores, preds = eval.test(trained_models, test_data)
        end_time = datetime.datetime.now()
        test_times[f"train"] = (end_time - start_time).seconds
        #collate our results
        preds = pd.DataFrame(preds)
        preds['set_id'] = ['test_res' for i in test_split]
        test_scores1 = {}
        test_scores1["MSE"] = {key: mean_squared_error(preds["y"],preds[key]) for key in  test_scores.keys()}
        test_scores1["R2"] = {key: r2_score(preds["y"],preds[key]) for key in  test_scores.keys()}
        if self.time:
            train_times['mean'] = np.mean([i for i in train_times.values()])
            test_times['mean'] = np.mean([i for i in test_times.values()])
            return test_scores1, preds, trained_models, train_times, test_times
        else:
            return test_scores1, preds, trained_models

class MangoesSplitter(SplitScheme):
    def __init__(self,*args,**kwargs):
        super(MangoesSplitter,self).__init__(*args,**kwargs)
        self.n_folds = 5
        self.cv_splitter = K2Fold(n_splits=self.n_folds,random_state=self.random_state)


    def evaluate(self, original_models, data:TabularDataset, eval:ModelScheme,pretrain=False,logger_name="",load_fun = None):
        logger = logging.getLogger(logger_name)
        logger.info("Running Cross Evaluation with {} folds".format(self.n_folds))

        # initialise datastructures for results
        model_states = {}
        preds = None
        total_scores = None
        scores = None
        #todo preds['y'] = []
        #todo preds['set_id'] = []

        model_states['init_state'] = {k:deepcopy(v.state()) for k,v in original_models.items()}

        train_times = {}
        test_times = {}
        #split our data
        train_ind1, val_test_ind1, test_ind1 = data.split_by_col(col = 'Set',train_key="Cal",val_key='Tuning',test_key='Val Ext')
        train_ind2 = np.union1d(train_ind1,val_test_ind1)
        #train_data,_,_ = data.split(train_ind2, None, test_ind1, preprocessing=None)
        unique_ids = data.meta_data['FruitID'][train_ind2].unique()

        #combine train_split and val_split
        ##, preprocessing=self.preprocessing)
        for fold, (train_ind_fruit, val_ind_fruit, test_ind_fruit) in enumerate(self.cv_splitter.split(unique_ids)):
            train_ind = data.meta_data[data.meta_data['FruitID'].isin(train_ind_fruit)].index
            val_ind = data.meta_data[data.meta_data['FruitID'].isin(val_ind_fruit)].index
            test_ind = data.meta_data[data.meta_data['FruitID'].isin(test_ind_fruit)].index
            train_data, val_data, test_data = data.split(train_ind, val_ind, test_ind, preprocessing=self.preprocessing)
            if load_fun is None:
                models = deepcopy(original_models)
            else:
                models = {name:load_fun(name,fold) for name,_ in original_models.items()}
            logger.info(f"-----------------------------------"
                             f"Fold {fold} - Train {len(train_data)} - Val {len(val_data)} - Test {len(test_data)}"
                             f"-----------------------------------")

            start_time = datetime.datetime.now()
            #pretrain
            if pretrain:
                #start = datetime.datetime.now()
                logger.info("Pretraining our models")
                eval.pretrain(models,train_data)
                #end = datetime.datetime.now()
                #self.logger.info(f"Pretraining took {end - start}")
            # train
            trained_models_fold, train_scores_fold = eval.train(models,train_data, val_data)
            end_time = datetime.datetime.now()
            train_times[f"fold_{fold}"] = (end_time-start_time).seconds
            model_states[f"fold_{fold}"] = {k:deepcopy(v.state()) for k,v in trained_models_fold.items()}

            #test
            start_time = datetime.datetime.now()
            test_scores_fold, preds_fold = eval.test(trained_models_fold,test_data)
            end_time = datetime.datetime.now()
            test_times[f"fold_{fold}"] = (end_time-start_time).seconds

            #record our predictions
            if preds is None:
                preds = {name:[] for name in preds_fold.keys()}
                preds['set_id'] = []
            for name in preds_fold.keys():
                preds[name] = preds[name] + preds_fold[name]
            preds['set_id'] = preds['set_id'] + [fold for i in val_ind]

            # record our scores
            if scores is None:
                scores = {}
                total_scores = {name:0 for name in test_scores_fold.keys()}
            scores[f"fold_{fold}"] = test_scores_fold
            total_scores = {name: total_scores[name] + test_scores_fold[name] for name in total_scores.keys()}
             
        #todo overall MSE and R^2
        scores["MSE"] = {key: mean_squared_error(preds["y"],preds[key]) for key in total_scores.keys()}
        scores["R2"] = {key: r2_score(preds["y"],preds[key]) for key in total_scores.keys()}
        #pprint(scores)

        preds = pd.DataFrame(preds)

        if self.time:
            train_times['mean'] = np.mean([i for i in train_times.values()])
            test_times['mean'] = np.mean([i for i in test_times.values()])
            return scores, preds, model_states, train_times,test_times
        else:
            return scores, preds, model_states

    def build(self, original_models, data: TabularDataset, eval: ModelScheme, pretrain=False, logger_name="",load_fun = None):
        logger = logging.getLogger(logger_name)
        train_times = {}
        test_times = {}

        train_ind, val_ind, test_ind = data.split_by_col(col = 'Set',train_key="Cal",val_key='Tuning',test_key='Val Ext')
        train_data, val_data, test_data = data.split(train_ind, val_ind, test_ind, preprocessing=self.preprocessing)
        logger.info(f"Building final model - Train {len(train_data)} - Test {len(test_data)}")

        if load_fun is None:
            models = deepcopy(original_models)
        else:
            models = {name: load_fun(name,"test_val") for name,_ in original_models.items()}

        start_time = datetime.datetime.now()
        if pretrain:
            #start = datetime.datetime.now()
            logger.info("Pretraining our models")
            eval.pretrain(models,train_data)
            #end = datetime.datetime.now()
            #self.logger.info(f"Pretraining took {end-start}")
        trained_models, train_scores = eval.train(models,train_data,val_data)
        #train
        end_time = datetime.datetime.now()
        train_times[f"train"] = (end_time - start_time).seconds
        #test
        start_time = datetime.datetime.now()
        test_scores, preds = eval.test(trained_models, test_data)
        end_time = datetime.datetime.now()
        test_times[f"train"] = (end_time - start_time).seconds
        #collate our results
        preds = pd.DataFrame(preds)
        preds['set_id'] = ['test_res' for i in test_ind]
        test_scores = {}
        test_scores["MSE"] = {key: mean_squared_error(preds["y"],preds[key]) for key in  train_scores.keys()}
        test_scores["R2"] = {key: r2_score(preds["y"],preds[key]) for key in  train_scores.keys()}
        if self.time:
            train_times['mean'] = np.mean([i for i in train_times.values()])
            test_times['mean'] = np.mean([i for i in test_times.values()])
            return test_scores, preds, models, train_times, test_times
        else:
            return test_scores, preds, models

class TrainTestEvaluation(SplitScheme):
    def __init__(self,*args,**kwargs):
        super(TrainTestEvaluation,self).__init__(*args,**kwargs)
        self.tt_splitter= train_test_split
        self.splitter = K2TrainTest(n_splits=5)

    def evaluate(self,original_models,data:TabularDataset,eval:ModelScheme,pretrain=False,logger_name=""):
        logger = logging.getLogger(logger_name)
        train_times = {}
        test_times = {}
        #split our data
        train_split1,test_split = self.tt_splitter(data,train_size=5/6,random_state=self.seed,shuffle=False)
        train_ind, val_ind, test_ind = self.splitter.split(train_split1)
        train_data, val_data, test_data = data.split(train_ind, val_ind, test_ind, preprocessing=self.preprocessing)
        models = deepcopy(original_models)
        #run our train method - this class doesn't want to know anything about these

        start_time = datetime.datetime.now()
        if pretrain:
            #start = datetime.datetime.now()
            logger.info("Pretraining our models")
            eval.pretrain(models,train_data)
            #end = datetime.datetime.now()
            #self.logger.info(f"Pretraining took {end-start}")
        trained_models, train_scores = eval.train(models,train_data,val_data)
        #train
        end_time = datetime.datetime.now()
        train_times[f"train"] = (end_time - start_time).seconds
        model_states['train']=trained_models
        #test
        start_time = datetime.datetime.now()
        test_scores, preds = eval.test(trained_models, test_data)
        end_time = datetime.datetime.now()
        test_times[f"train"] = (end_time - start_time).seconds
        #collate our results
        preds = pd.DataFrame(preds)
        preds['set_id'] = ['test' for i in val_ind]
        test_scores = {"mean": test_scores, "test_split": test_scores}
        if self.time:
            train_times['mean'] = np.mean([i for i in train_times.values()])
            test_times['mean'] = np.mean([i for i in test_times.values()])
            return test_scores, preds, model_states, train_times, test_times
        else:
            return test_scores, preds, model_states

    def build(self,models,data:TabularDataset,eval:ModelScheme,pretrain=False,logger_name=""):
        train_times = {}
        test_times = {}
        logger = logging.getLogger(logger_name)
        logger.info("Running a Train Test Evaluation")
        #split our data
        train_split1,test_ind = self.tt_splitter(data,train_size=5/6,random_state=self.seed,shuffle=False)
        train_split, val_ind, test_split = self.splitter.split(train_split1)
        train_ind = np.union(train_split,test_split)

        train_data, val_data, test_data = data.split(train_ind, val_ind, test_ind, preprocessing=self.preprocessing)
        #run our train method - this class doesn't want to know anything about these

        start_time = datetime.datetime.now()
        if pretrain:
            #start = datetime.datetime.now()
            logger.info("Pretraining our models")
            eval.pretrain(models,train_data)
            #end = datetime.datetime.now()
            #self.logger.info(f"Pretraining took {end-start}")
        trained_models, train_scores = eval.train(models,train_data,val_data)
        #train
        end_time = datetime.datetime.now()
        train_times[f"train"] = (end_time - start_time).seconds
        #test
        start_time = datetime.datetime.now()
        test_scores, preds = eval.test(trained_models, test_data)
        end_time = datetime.datetime.now()
        test_times[f"train"] = (end_time - start_time).seconds
        #collate our results
        preds = pd.DataFrame(preds)
        preds['set_id'] = ['test' for i in test_ind]
        test_scores = {"mean": test_scores, "test_split": test_scores}
        if self.time:
            train_times['mean'] = np.mean([i for i in train_times.values()])
            test_times['mean'] = np.mean([i for i in test_times.values()])
            return test_scores, preds, models, train_times, test_times
        else:
            return test_scores, preds, models


def train_batch(X, y, models, opts, loss_eval, loss_fun,time=False):
    """
    function to run a train batch
    :param X:
    :param y:
    :param model: dict of name,models, where model is a pytorch model
    :param opt:
    :param loss_eval:
    :param loss:
    :return:
    """
    if time:
        start = datetime.datetime.now()
        results = train_batch(X, y, models, opts, loss_eval, loss_fun, False)
        end = datetime.datetime.now()
        return results,end-start

    losses = {}
    for name,model in models.items():

        model.train()
        opts[name].zero_grad()
        pred = model(X)

        loss = loss_eval(X,y,pred,loss_fun)

        #setup training
        loss.backward()
        losses[name] = loss.detach().cpu().tolist()

        opts[name].step()
        #clear up any gpu memory, just in case

        del pred,loss
        torch.cuda.empty_cache()

    #todo fix up
    return losses


def adaptive_simple(schedulers=None, opts=None, models=None, scores=None,name=""):
    # adapt our lr across space (our set of random models) and across time (
    # todo - relationship of score~lr.
    # make a model that

    # we want to consider that we are also searching across each instance
    #
    X, y = zip(*[(opts[name].param_groups[0]['lr'], scores[name]) for name in models.keys()])
    ix = np.argmin(y)
    best_lr = X[ix]
    current_lr = opts[name].param_groups[0]['lr']

    if best_lr > current_lr: #need to increase lr
        #models[name].reset()
        opts[name].param_groups[0]['lr'] /= random.uniform(0.1, 0.9)
    elif best_lr < current_lr: # need to decrease lr
        #models[name].reset()
        opts[name].param_groups[0]['lr'] *= random.uniform(0.1, 0.9)
    else: #they are equal
        pass

    if min(scores.values())*100 < scores[name]:
        models[name].reset()
    #move in direction



def iterate_schedulers(schedulers=None, opts=None, models=None, scores=None):
    for name, scheduler in schedulers.items():
        value = scores[name]
        if np.isnan(value):
            models[name].reset()
            current_val = math.log10(opts[name].param_groups[0]['lr'])
            random_val = random.uniform(0.5, 3.5)
            opts[name].param_groups[0]['lr'] = pow(10, current_val - random_val)
        elif value > 10 * min(scores.values()):
           adaptive_simple(schedulers=schedulers, opts=opts, models=models, scores=scores,name=name)
            # we can set at best lr, +- a range
        else:
            if scheduler is None:
                pass
            elif type(scheduler) is optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(value)
            else:
                scheduler.step()

def adaptive_gpr_lr_update(schedulers=None, opts=None, models=None, scores=None):
    #adapt our lr across space (our set of random models) and across time (
    #todo - relationship of score~lr.
    #make a model that

    #we want to
    adaptive_model = GaussianProcessRegressor()
    X,y = zip(*[(opts[name].param_groups[0]['lr'],scores[name]) for name in models.keys()])
    adaptive_model.fit(X, y) # X is lr,

    def sample_X():
        return pow(10, -(random.uniform(-1, 10)))

    def sample_y(X):
        return adaptive_model.predict(X)

    def opt_acquisition(n=100):
        # random search, generate random samples
        X_samples = [sample_X() for i in range(0,n)]
        scores = [sample_y(X_sample) for X_sample in X_samples]
        # locate the index of the largest scores
        ix = np.argmax(scores)
        return X_samples[ix]

    #sample based on good territory


def heuristic_lr_update(schedulers = None, opts = None, models=None, scores=None,name=""):
        models[name].reset()
        opts[name].param_groups[0]['lr'] *= 0.5


def test_batch(X, y, models, opts, loss_eval,loss_fun,time=False):
    """
    function to run a test batch
    :param X:
    :param y:
    :param name:
    :param model:
    :param opt:
    :param loss_eval:
    :param loss:
    :return:
    """

    if time:
        start = datetime.datetime.now()
        results = test_batch(X, y, models, opts, loss_eval, loss_fun, False)
        end = datetime.datetime.now()
        return results,end-start

    losses = {}
    preds = {}
    with torch.no_grad():
        #setup testing
        for name,model in models.items():
            model.eval()
            pred = model(X)
            loss = loss_eval(X, y, pred, loss_fun)

            losses[name]=loss.detach().cpu().tolist()
            preds[name]= pred.detach().cpu().tolist()

            torch.cuda.empty_cache()
            del pred, loss
    return preds, losses