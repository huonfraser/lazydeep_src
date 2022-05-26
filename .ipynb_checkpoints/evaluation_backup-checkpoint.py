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
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

import argparse
import time,datetime
import logging
import jsonpickle
from copy import deepcopy
from math import inf
from math import sqrt

from splitters import *
from utils import *
from lazydeep import *
import math
from math import log10, inf
from sk_models import StandardScaler


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


def lr_finder(train_data,models,configs, preprocessing = StandardScaler,
              start_lr=1e-7, end_lr=10, num_it:int=100, wd:float=None,criterion = nn.MSELoss(), beta = 0.7):
    #sort out data
    preprocessing.fit(train_data)
    train_data = preprocessing.transform(train_data)
    train_set = utils.TabularDataset(data=train_data, cat_cols=[])
    test_set = utils.TabularDataset(data=train_data, cat_cols=[])
    train_loader = DataLoader(train_set,shuffle=True, num_workers=0,batch_size=32)

    num_it:max(num_it,len(train_loader))


    test_X = None
    test_y = None
    #setup starting parameters
    mult = (end_lr / start_lr) ** (1/num_it)
    lr = start_lr

    opts = {}
    for i, (name, model) in enumerate(models.items()):
        config = configs[name]
        opt = config.opt(model.parameters(), lr=lr)
        opts[name] = opt

    #setup datastucture to record results
    best_losses = {name:inf for name in models.keys()}
    batch_num = 0
    losses = {name:[] for name in models.keys()}
    avg_losses = {name:0 for name in models.keys()}
    #cont_bool = {name:True for name in models.keys()}
    log_lrs = []

    #todo pretain a few batches at the small lr
    while True:
        for X,y in train_loader:
            batch_num += 1
            X = X.cpu().float()
            y = y.cpu().float()

            #As before, get the loss for this mini-batch of inputs/outputs

            batch_losses = train_batch(X,y,models,opts,loss_target,criterion)
            avg_losses = {name: beta * avg_losses[name] + (1 - beta) * loss for name, loss in batch_losses.items()}
            smoothed_losses = {name: avg_loss / (1 - beta ** batch_num) for name, avg_loss in avg_losses.items()}

            #print(batch_losses)
            #Compute the smoothed loss
            #
            #

            #add nans

            #Record the best loss
            for name, smoothed_loss in smoothed_losses.items():
                if smoothed_loss < best_losses[name] or batch_num == 1:
                    best_losses[name] = smoothed_loss
                losses[name].append(smoothed_loss)

            log_lrs.append(log10(lr))


            #Update the lr for the next step
            lr *= mult
            for name, opt in opts.items():
                opt.param_groups[0]['lr'] = lr
           # print(batch_num)

            if batch_num > num_it:
                return log_lrs, losses




class Evaluation():

    def __init__(self, loss_eval = loss_target,
                 loss_fun_sk = mean_squared_error,
                 loss_fun_torch = nn.MSELoss(),
                 preprocessing = None,
                 splitter = None,
                 tensorboard = None,
                 logger = None
                 ):
        """
        Define the structure of our
        :param loss_eval: how to evaluate our loss function
        :param loss_fun_sk: loss function for
        :param loss_fun_torch:
        """

        self.loss_eval = loss_eval
        self.loss_fun_sk = loss_fun_sk
        self.loss_fun_torch = loss_fun_torch
        self.preprocess = preprocessing
        self.splitter = splitter
        self.tensorboard = tensorboard

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

    def evaluate_pls(self,data,models):
        pass

    def _evaluate_pls(self,models,train_data,test_data,n_components=5,scale=True):
        """
        Internal method to apply pls feature extraction before a bunch of models

        :param models:
        :param train_data:
        :param test_data:
        :param n_components:
        :param scale:
        :return: a dict of scores and one of predictiosn
        """

        #todo will this work
        if self.preprocess is not None:
            self.preprocess.fit(train_data)
            train_data = self.preprocess.transform(train_data)
            test_data = self.preprocess.transform(test_data)

        train_X_g = (X for X, y in train_data)
        train_y_g = (y for X, y in train_data)
        test_X_g = (X for X, y in test_data)
        test_y_g = (y for X, y in test_data)

        train_X = np.asarray([X for X in train_X_g])
        test_X = np.asarray([X for X in test_X_g])
        train_y = np.asarray([y for y in train_y_g])
        test_y = np.asarray([y for y in test_y_g])

        nrow, ncol = train_X.shape
        n_features = ncol

        preds = {name: [] for name in models.keys()}
        preds['y'] = test_y.detach().cpu().tolist()

        # fit pls
        if n_components is None:
            n_components = n_features
        pls = PLSRegression(n_components=n_components, scale=scale)
        pls.fit(train_X, train_y)

        # transform data
        X_train_c, y_train_c = pls.transform(train_X, train_y)
        X_test_c, y_test_c = pls.transform(test_X, test_y)

        fold_scores = {}
        for name, model in models.items():
            # fit model
            model.fit(X_train_c, train_y)

            # test model
            y_pred = model.predict(X_test_c)
            # print(y_pred)
            score = self.loss_fun_sk(test_y, y_pred)

            fold_scores[name] = score
            preds[name] = preds[name] + y_pred.detach().cpu().tolist()
        return fold_scores, preds

    def evaluate_sk(self, data, models, configs=None, fixed_hyperparams=None,pretrained=True):
        pass

    def evaluate_torch(self, data, models, configs, fixed_hyperparams):
        """

        :param data:
        :param models:
        :param configs:
        :param fixed_hyperparams:
        :return:
            1) test set scores: dict of {model_name, int}
            2) model states:  {train_set_name,{model_name,state_dict}}
        """
        pass

    def train_torch(self, train_data, models, configs,fixed_hyperparams=None, val_data=None):
        """
        Train a pytorch type model
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        train_start = datetime.datetime.now()

        bs = fixed_hyperparams['bs']
        loss = fixed_hyperparams['loss']
        epochs = fixed_hyperparams['epochs']


        #fit preprocessing if required]
        #todo write a method to extract preprocesed data , maybe make tabular dataset an iteratable dataset

        #if not self.preprocess is None:
        #    pp_fit = [i for i,j in train_data]
        #    #pp_fit = pd.DataFrame(pp_fit).to_numpy()
        #    self.preprocess.fit(pp_fit)
            #train_data = preprocess_subset()
            #test_data =
        #print(train_data)
        #val_loader = DataLoader(val_data, batch_size=bs, shuffle=False)
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)

        # setup optimisers and schedulers
        opts = {}
        schedulers = {}
        for i, (name, model) in enumerate(models.items()):
            config = configs[name]
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
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,20)
                schedulers[name] = scheduler
            else:
                print(config.lr_update.__name__)


        #record best model and state
        best_losses = {name: inf for name in models.keys()}
        best_models = {name: deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()}) for name, model in models.items()}

        self.logger.info("Training extractors on {} instances, validating on {} instances, for {} epochs".format(
            len(train_data), len(val_data), epochs))

        n_train_batches = 0

        for epoch in range(0, epochs):
            self.logger.info("\n--- EPOCH {}---".format(epoch))
            epoch_start = datetime.datetime.now()

            #train
            epoch_train_total = {name: 0 for name in models.keys()}
            n_observed = 0
            for batch, (X, y) in enumerate(train_loader):
                #if self.preprocess is not None:
                #    X = X.detach().cpu().float()
                #    X = torch.Tensor(self.preprocess.transform(X)).to(device).float()
                #else:

                X = X.to(device).float()
                y = y.to(device).float()

                bs = y.shape[0]
                n_observed += bs

                batch_losses = train_batch(X, y, models, opts, self.loss_eval, self.loss_fun_torch)
                epoch_train_total = {name: epoch_train_total[name] + np.sum(batch_losses[name] * bs) for name in
                                       models.keys()}
                #if not self.tensorboard is None:
                #    self.tensorboard.add_scalars("batch/train/loss", batch_losses, n_train_batches)
                n_train_batches +=1
                del X,y
                torch.cuda.empty_cache()

                # todo should reset models here
            from math import log10
            #log training
            epoch_train_scores = {name: value / n_observed for name, value in epoch_train_total.items()}
            self.logger.info("Extractor Train Losses are " + ",".join(
                str(i) + ":" + str(round(j, 4)) + "(" + str(round(log10(opts[i].param_groups[0]['lr']),6)) + ")" for (i, j) in epoch_train_scores.items()))
            if not self.tensorboard is None:
                    self.tensorboard.add_scalars("epoch/train/loss", epoch_train_scores, epoch)

            epoch_middle = datetime.datetime.now()

            #validate
            epoch_val_scores, epoch_val_preds = self.test_torch(val_data,models,configs,fixed_hyperparams,test_type = "val")
            if not self.tensorboard is None:
                self.tensorboard.add_scalars("epoch/val/loss", epoch_train_scores, epoch)

            # run lr schedulers. Either restart models or run schedulers
            iterate_schedulers(schedulers=schedulers, opts=opts, models=models, scores=epoch_train_scores)

            # store current best model
            for name, model in models.items():
                if epoch_val_scores[name] < best_losses[name]:
                    best_losses[name] = epoch_val_scores[name]
                    best_models[name] = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

            epoch_end = datetime.datetime.now()
            epoch_diff = (epoch_end - epoch_start)

            del epoch_val_scores,epoch_val_preds, epoch_train_total,batch_losses
            self.logger.info("Epoch {} finished in {} ".format(epoch, epoch_diff))
        self.logger.info("\n-----------")
        self.logger.info("Finished training extractors with a best validation loss of " + ",".join(
            str(i) + ":" + str(round(j, 4)) for (i, j) in best_losses.items()))

        # return best models along with validation results

        train_end = datetime.datetime.now()
        train_diff = train_end - train_start
        self.logger.info("Training took {}".format(train_diff))
        #todo final learning rates

        return best_models, best_losses

    def test_torch(self, data, models, configs=None, fixed_hyperparams=None, test_type="test"):
        """'=
        Test a pytorch type model
        Log each batch to pytorch
        """
        bs = fixed_hyperparams['bs']
        loss = fixed_hyperparams['loss']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
                X = X.to(device).float()
                y = y.to(device).float()
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

    def test_sk(self,data, models,fixed_hyperparams = None):
        """
        Test sklearn-type models
        :param data:
        :param models:
        :param fixed_hyperparams:
        :param preprocess:
        :param log_file:
        :return:
        """

        test_start = datetime.datetime.now()
        data = self.preprocess.transform(data)

        nrow, ncol = data.shape
        # utils.log("Testing final models on {} instances".format(nrow),log_file)

        y = data["target"]
        X = data.drop(columns="target")

        results = utils.PredictorResults()

        for i, (name, model) in enumerate(models.items()):
            preds = model.predict(X)
            accuracy = mean_squared_error(y, preds)

            results.add_results([j for j in range(0,nrow)],name,y,preds,accuracy)
        self.logger.info("knn Losses are ".format(nrow)+",".join(str(i)+":"+str(round(j,4)) for i,j in results.score().items()))

        test_end = datetime.datetime.now()
        test_diff = test_end-test_start
        #utils.log("Finished testing predictors in {}".format(test_diff))
        return results

    def train_sk(self,data:pd.DataFrame, models,fixed_hyperparams = None):
        """
        Train sklearn-type models
        :param data: pd dataframe containing training split
        :param model: a dict of initialised lazy deep models
        :return:
        """
        train_start = datetime.datetime.now()
        #withold a validation split
        self.preprocess.fit(data)
        data = self.preprocess.transform(data)

        y = data["target"]
        X = data.drop(columns="target")

        #utils.log("".format(nrow,log_file))
        results = {}
        for i, (name, model) in enumerate(models.items()):
            result = model.fit(X,y)

        train_end = datetime.datetime.now()
        train_diff = train_end-train_start
        #utils.log("Trained predictor on {} instances in {}".format(nrow,train_diff),log_file=log_file)
        return models,results

class CrossValEvaluation(Evaluation):

    def __init__(self,n_folds = 5,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_folds = n_folds
        if self.splitter is None:
            self.splitter = K2Fold(5)

    def evaluate_pls(self,data:TabularDataset,models,n_components=5,scale = True):

        self.logger.info("Evaluating PLS")
        scores = {}
        total_scores = {name: 0 for name in models.keys()}

        preds = {name: [] for name in models.keys()}
        preds['y'] = []
        preds['set_id'] = []

        for fold, (train_ind, val_ind, test_ind) in enumerate(self.splitter.split(data)):
            self.logger.info(f"Training PLS - Fold {fold} -on {len(train_ind)}, Testing on {len(test_ind)}")
            train_data, val_data, test_data = data.split(train_ind,val_ind,test_ind,preprocessing=self.preprocess)

            fold_scores,fold_preds = self._evaluate_pls(models,train_data,val_data,n_components=n_components,scale=scale)
            #append preds
            fold_preds['set_id']=[f"fold_{fold}"]*len(fold_preds["y"])
            preds = {key:preds[key]+fold_preds[key] for key in preds.keys()}

            #append scores
            scores[f"fold_{fold}"] = fold_scores
            total_scores = {key:total_scores[key]+fold_scores[key] for key,value in fold_scores.items()}

        #calculate aggregates and return
        scores["mean"] = {name:score/self.n_folds for name, score in total_scores.items()}
        preds = pd.DataFrame(preds)
        return scores, preds


    def evaluate_torch(self, data:TabularDataset, models, configs, fixed_hyperparams):
        self.logger.info("Running Cross Evaluation with {} folds".format(self.n_folds))

        #initialise datastructures for results
        model_states = {}
        total_scores = {name: 0 for name in models.keys()}
        scores = {}
        preds = {name: [] for name in models.keys()}
        preds['y'] = []
        preds['set_id'] = []
        for fold, (train_ind,val_ind, test_ind) in enumerate(self.splitter.split(data)):
            #reset
            [model.reset() for name, model in models.items()]

            # take split
            train_data, val_data, test_data = data.split(train_ind, val_ind, test_ind, preprocessing=self.preprocess)
            self.logger.info(f"-----------------------------------"
                             f"Fold {fold}"
                             f"-----------------------------------")
            #train
            fold_model_states, fold_train_results = self.train_torch(train_data, models, configs,
                                                                     fixed_hyperparams=fixed_hyperparams,val_data=val_data)
            model_states[f"cv_{fold}"]=fold_model_states

            #test
            fold_test_results,fold_preds = self.test_torch(test_data, models, configs,
                                                           fixed_hyperparams=fixed_hyperparams)
            for name in fold_preds.keys():
                preds[name] = preds[name] + fold_preds[name]
            preds['set_id'] = preds['set_id'] + [fold for i in test_ind]
            scores[f"fold_{fold}"] = fold_test_results

            total_scores = {name: total_scores[name] + fold_test_results[name] for name in models.keys()}
            del train_data, val_data, test_data,fold_model_states, fold_train_results,fold_test_results,fold_preds
            torch.cuda.empty_cache()
        #print(total_scores)
        scores["mean"] = {name: score/self.n_folds for name, score in total_scores.items()}
        #print(scores)
        preds = pd.DataFrame(preds)


        return scores, preds, model_states


    def evaluate_sk(self, data, models, configs, fixed_hyperparams,pretrained=True):
        pass


class TrainTestEvaluation(Evaluation):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.splitter is None:
            self.splitter = TrainTestValSplit(train_prop=0.6, test_prop=0.2, val_prop=0.2)

    def evaluate_pls(self, data, models,n_components=None,scale=True):
        train_ind, val_ind, test_ind = self.splitter.split(data)
        train_data, val_data, test_data = data.split(train_ind, val_ind, test_ind, preprocessing=self.preprocess)
        fold_scores,preds = self._evaluate_pls(models,train_data,test_data,n_components=n_components,scale=True)

        #append preds
        preds['set_id'] = ['test' for i in test_ind]
        preds = pd.DataFrame(preds)

        # append scores
        scores={}
        scores[f"test"] = fold_scores
        scores["mean"] =  fold_scores

        return scores, preds

    def evaluate_torch(self, data, models, configs, fixed_hyperparams,test_data = None, val_data=None):
        self.logger.info("Running a Train Test Evaluation")
        train_ind, val_ind,test_ind = self.splitter.split(data)
        train_data, val_data, test_data = data.split(train_ind, val_ind, test_ind, preprocessing=self.preprocess)

        for name, model in models.items():
            model.reset()

        trained_models, train_scores = self.train_torch(train_data, models, configs, fixed_hyperparams=fixed_hyperparams,val_data=val_data)
        test_scores, preds = self.test_torch(test_data, models, configs, fixed_hyperparams=fixed_hyperparams)

        preds = pd.DataFrame(preds)
        preds['set_id'] = ['test' for i in test_ind]
        test_scores = {"mean":test_scores, "test_split":test_scores}
        return test_scores, preds, {"train": {k: v for k, v in trained_models.items()}}

    def evaluate_sk(self, data, models, configs, fixed_hyperparams, pretrained=True):
        pass

class PrefitTTEvaluation(TrainTestEvaluation):
    """
    Class runs evaluation -after calling the fit method on the deep networks
    """


def train_batch(X, y, models, opts, loss_eval, loss_fun):
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
    losses = {}
    for name,model in models.items():

        model.train()
        opts[name].zero_grad()
        pred = model(X)


        loss = loss_eval(X,y,pred,loss_fun)
        del pred
        torch.cuda.empty_cache()


        #setup training
        loss.backward()
        losses[name] = loss.detach().cpu().tolist()



        opts[name].step()


        del loss
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


def test_batch(X, y, models, opts, loss_eval,loss_fun):
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