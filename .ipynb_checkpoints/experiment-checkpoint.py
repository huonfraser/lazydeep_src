from torch import nn, save, optim
from torch import optim

from random import sample
from autoencoder import *
from evaluation import *
from splitters import *
from pathlib import Path
import pandas as pd

from plot import *

from lazydeep import *
from deep_net import *
from plot import *
from sklearn.metrics import  mean_squared_error, r2_score

from os import mkdir, path
import logging
import sys

from configurations import RandomConfigGen, Configuration
import scipy
import matplotlib.pyplot as plt
from pipeline import Learner, Pipeline

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed + 1)
    np.random.seed(seed + 2)


def save_models(models,configs,log_dir,prefix=""):
    # save models and configs
    for name, model in models.items():
        if log_dir is not None:
            output_name = log_dir/"models"
            if not output_name.exists():
                output_name.mkdir()
            output_name = log_dir/"models"/"{}".format(name)
            if not output_name.exists():
                output_name.mkdir()
            # save model
            save(model, log_dir/"models"/f"{name}"/f"{prefix}_model")
            # save configs
            configs[name].save(log_dir/"models"/f"{name}"/f"{prefix}_config")

    pass

def write_summary_head(seed,fixed_hyperparams,prefix=""):
    summary_logger = logging.getLogger(prefix+"summary")
    summary_logger.info("Starting Experiment")
    summary_logger.info("Seed: {}".format(seed))
    summary_logger.info("bs: {}".format(fixed_hyperparams['bs']))  # "'bs': 32,'loss': nn.MSELoss,epochs: 100}
    # log("loss {}".format(fixed_hyperparams['loss']), summary_file)
    summary_logger.info("epochs: {}".format(fixed_hyperparams['epochs']))
    summary_logger.info("--------------------")

def write_summary(time,models,loss_scores,prefix=""):
    summary_logger = logging.getLogger(prefix+"summary")
    summary_logger.info("Experiments took {}".format(time))

    #depth = {name: model.n_layers for name, model in models.items()}
    #n_features = {name: model.n_features for name, model in models.items()}

    # rank order models and create a master results file
    summary_logger.info("Finished Random Deep Search")
    summary_logger.info("---Loss results---")
    #summary_logger.info("Rank - ID - n_layers - n_features - score")


    for i, (key, v) in enumerate(sorted(loss_scores["MSE"].items(), key=lambda x: x[1])):
        results = []
        for k,v2 in loss_scores.items():
            results.append((k,v2[key]))
        #for name in loss_scores.keys():
        #    if not name == 'mean':
               # results.append((name,loss_scores[name][key]))

        summary_logger.info(f"{i} - {key} - " + ",".join(str(j)+':'+str(round(k,4)) for j,k in results ))

       #todo write r^2 summary
                           # f"{[loss_scores['mean'][key],loss_scores['fold_0'][key],loss_scores['fold_1'][key],loss_scores['fold_2'][key],loss_scores['fold_3'][key],loss_scores['fold_4'][key]]}")

def save_pp(pp,log_dir,prefix=""):
    from pipeline import Learner
    output_name = log_dir/"preprocessing" 
    if not output_name.exists():
                output_name.mkdir()
    for name, model in pp.items():
        Pipeline().save_state(model, output_name / f"{prefix}_{name}")


def save_results(model_states,pred_db,configs,loss_scores,log_dir,tb,prefix=""):
        # save all 5 model states
    for name, nested in model_states.items():
        for model, state_dict in nested.items():
            torch.save(state_dict, log_dir / "models" / f"{model}" / f"{prefix}_{name}")

    if not tb is None:
        for name, config in configs.items():
            hparams = config.to_dict()
            tb.add_hparams(hparams, {"MSE": loss_scores["MSE"][name]}, run_name=name)

    pred_db.to_csv(log_dir / (f"{prefix}_predictions" + ".csv"), index=False)

def save_pred_plots(pred_db,models,log_dir,prefix=""):
    pred_db = pred_db.dropna()
    if len(pred_db) > 0:
        for name in pred_db.columns:
            if (not name == "y") and (not name == "set_id"):
                dir = log_dir / f"models"/f"{name}"
                if not dir.exists():
                    dir.mkdir(parents=True)
                # plot predictions
                fig, ax = scatter_plot(pred_db,name,"y",color_col="set_id",title=f"Predictions for {name}")


                plt.savefig(dir/f"{prefix}_predictions.png",bbox_inches='tight')
                plt.close()
                # plt.show()

                fig, ax = residual_plot(pred_db, name, "y", color_col="set_id", title=f"Residuals for {name}")
                plt.savefig(dir/f"{prefix}_residuals.png", bbox_inches='tight')
                plt.close()

def run_experiment(models,configs,fixed_hyperparams,data,
                   splitter = None,
                   tb=None,
                   log_dir:Path = None,
                   seed = 1,
                   eval = "tt",
                   preprocessing = None) -> (dict,pd.DataFrame,dict):
    set_seed(seed)
    write_summary_head(seed,fixed_hyperparams)
    save_models(models,configs,log_dir)

    start = datetime.datetime.now()
    if eval == "cv":
        eval = CrossValEvaluation(preprocessing=preprocessing,tensorboard=tb, splitter = splitter)
        loss_scores, pred_db, model_states = eval.evaluate_torch(data, models, configs,
                                                                 fixed_hyperparams=fixed_hyperparams)
    else:
        eval = TrainTestEvaluation(preprocessing=preprocessing,tensorboard=tb, splitter = splitter)
        loss_scores, pred_db, model_states = eval.evaluate_torch(data, models, configs,
                                                                 fixed_hyperparams=fixed_hyperparams)

    save_results(model_states, pred_db, configs, loss_scores, log_dir,tb)

    end = datetime.datetime.now()
    diff = end - start
    write_summary(diff, models, loss_scores)
    save_pred_plots(pred_db, models, log_dir)


class RandomDeepSearch():
    """
    We set up a
    """
    def __init__(self, input = None, num = 100, epochs = 100, seed=1, log_dir = ""):

        #define fixed hyperparametesr
        fixed_hyperparams = {'bs': 32,'loss': nn.MSELoss(),'epochs': epochs}
        config_gen = RandomConfigGen(lr= [1e-3,1e-4,1e-5],allow_increase_size=False)
        config_gen.save(log_dir/'config_gen.txt')
        #define hyperparameter generator

        nrow, ncol = data.shape
        n_features = ncol - 1


        #setup instances
        configs = {}
        models = {}
        for i in range(0,num):
            # generate random parameters
            config = config_gen.sample()
            configs['random_{}'.format(i)] = config
            models['random_{}'.format(i)] = RandomNet(input_size=n_features,n_layers=config.n_layers,act_function=config.act_function)
            #instances.append((i,models,config,output_dir))
        scores, predictions, model_states= run_experiment(models,configs,fixed_hyperparams,data,log_dir= log_dir, seed = seed)
        #todo regression of hyperparameters


if __name__ == "__main__":
    iris_file = 'D:/workspace/lazydeep/data/batch_data/iris.csv'
    abalone_file = 'D:/workspace/lazydeep/data/stream_data/regression/abalone.csv'
    a_al_rt_file = 'D:/workspace/lazydeep/data/soil_data/A_AL_RT.csv'
    a_c_file = '/soil_data/A_C_OF_SIWARE.csv'

    data = pd.read_csv(a_c_file)




    RandomDeepSearch(input=data, num=100, epochs=100,seed=1, output_dir='D:/workspace/lazydeep/outputs/next')