import torch
from torch import nn
from torch import optim
import torch.functional as F
from torch.utils.data import DataLoader, Dataset

from copy import deepcopy
import jsonpickle

import random as r
import numpy as np
from dataclasses import dataclass

@dataclass
class Configuration:
    """
        Container class for variables
    """
    opt: int = optim.SGD,
    lr: float = 1e-6,
    lr_update: int = None,
    n_layers: int = 5,
    act_function: int = nn.ReLU,
    allow_increase_size: bool = False,
    n_features: int = None,
    dropout: bool = True,
    batch_norm: bool = False

    def to_dict(self):
        if self.lr_update is None:
            lr_temp = "None"
        else:
            lr_temp = str(self.lr_update)

        dict_form = {"opt":self.opt.__name__,
                "lr":self.lr,
                "lr_update":lr_temp,
                "n_layers":self.n_layers,
                "n_features":self.n_features,
                "act_function":self.act_function.__name__,
                "allow_increase_size":self.allow_increase_size}
        return dict_form

    def set_lr(self,lr):
        c = deepcopy(self)
        c.lr=lr
        return c

    def set_lr_update(self, lr_update):
        c = deepcopy(self)
        c.lr_update = lr_update
        return c

    def save(self,fname):
        json = jsonpickle.encode(self)
        with open(fname,'w+') as file:
            file.write(json)
        #todo save to fname

    @staticmethod
    def load(fname):
        with open(fname,'r') as file:
            text = file.read()
            return jsonpickle.decode(text)

class RandomConfigGen():

    def __init__(self,
                 opt = [optim.SGD, optim.Adam, optim.AdamW],
                 lr = (1, 10),
                 n_layers= [2, 3, 4, 5],
                 lr_update = [None, torch.optim.lr_scheduler.ReduceLROnPlateau],
                 act_function = [nn.ReLU, nn.RReLU, nn.LeakyReLU],
                 allow_increase_size = False,
                 dropout = [True, False],
                 batch_norm = [True, False],
                 n_features = 100):
        self.opt = opt
        self.n_layers = n_layers
        self.lr = lr
        self.lr_update = lr_update
        self.act_function = act_function
        self.allow_increase_size = allow_increase_size
        self.n_features = n_features
        self.dropout = dropout
        self.batch_norm = batch_norm


        #self.n_params = len(self.opt)+len(self.lr_update)+len(self.act_function)+2

    def sample(self):
        return Configuration(opt = r.sample(self.opt,1)[0], #sample(self.opt,1)[0],
                             lr = pow(10,-r.uniform(self.lr[0],self.lr[1])),
                             lr_update = r.sample(self.lr_update,1)[0],
                             n_layers = r.sample(self.n_layers,1)[0],
                             act_function= r.sample(self.act_function,1)[0],
                             allow_increase_size=self.allow_increase_size,
                             n_features = r.randint(1,self.n_features),
                             dropout = r.sample(self.dropout,1)[0],
                             batch_norm = r.sample(self.batch_norm,1)[0]
                             )

    def save(self,fname):
        json = jsonpickle.encode(self)
        with open(fname, 'w+') as file:
            file.write(json)

    def load(self,file):
        pass


    def to_array(self,config):
        array= np.zeroes((self.n_params,1))

        if config.opt == optim.SGD:
            array[0] = 1
        else:
            array[1] = 1

        if config.lr_update is None:
            array[2] = 1
        else:
            array[3] = 1

        if config.act_functions is nn.ReLU:
            array[4] = 1
        elif config.act_functions is nn.RReLU:
            array[5] = 1
        else:
            array[6] = 1

        array[7] = config.n_layers
        array[8] = config.lr

    def from_array(self,X):

        if X[0] == 1:
            opt = optim.SGD
        else:
            opt = optim.Adam

        if X[2] == 1:
            lr_update = None
        else:
            lr_update = optim.lr_scheduler.ReduceLROnPlateau

        if X[4] == 1:
            act_function = nn.ReLU
        elif X[5] == 1:
            act_function = nn.RReLU
        else:
            act_function =  nn.LeakyReLU

        n_layers = X[7]
        lr= X[8]

        return Configuration(opt = opt,
                             lr = lr,
                             lr_update = lr_update,
                             n_layers =  n_layers,
                             act_function= act_function
                             )



