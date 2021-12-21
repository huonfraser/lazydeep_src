from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.functional as F
from sklearn.model_selection import train_test_split, KFold
from copy import deepcopy
import numpy as np
from scipy.io import arff
from torch.utils.data import DataLoader, Dataset
import pandas as pd

import utils
import experiment
import argparse
from IPython.display import clear_output

from sklearn.model_selection import KFold
import time,datetime
import jsonpickle
from math import inf,sqrt

import random

from deep_net import DeepNet


class AutoEncoder(nn.Module):

    def __init__(self,  regularisation = 'none'):
        """
        :param encode:
        :param bottle:
        :param decode:
        :param weights: {"default","ones","zeroes","uniform"}
        :param default_activation:
        """
        super(AutoEncoder, self).__init__()
        self.encode=nn.Sequential()
        self.bottle=nn.Sequential()
        self.decode=nn.Sequential()

        self.reset_weights = None

    def forward(self, x,*args,**kargs):
        return self.decode(self.bottle(self.encode(x)))

    def compress(self, x):
        return self.bottle(self.encode(x))

    def decompress(self, x):
        return self.decode(x)

    def extract_compress(self):
        new_model = nn.Sequential()
        new_model.add(self.encode_layer)
        return new_model

    def extract_decompress(self):
        new_model = nn.Sequential(self.decode_layer)
        return new_model

    def init_weights(self, m):
        if type(m) == nn.Linear:

            if self.init_rule_weight == 'none':
                stdv = 1. / sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv,stdv)

            if m.bias is not None:
                if self.init_rule_bias == 'none':
                    m.bias.data.uniform_(-stdv,stdv)

            #torch.nn.init.uniform_(m.weight,0,1)
            #torch.nn.init.ones_(m.weight)
            #torch.nn.init.zeros_(m.weight)
            #m.bias.data.fill_(0)
            pass

    def reset(self):
        self.parameters = self.reset_weights

    def setup(self):
        self.reseed()

        self.reset_weights=self.parameters

    def reseed(self):
        self.apply(self.init_weights)

class DeepIdentityEncoder(DeepNet):
    def __init__(self,input_size=64, n_layers=5,act_fun=nn.ReLU,*args,**kwargs):
        super(DeepIdentityEncoder, self).__init__(*args,**kwargs)

        self.n_layers = n_layers
        self.act_fun = act_fun

        #todo bottleneck and batch norm
        encode_layers = OrderedDict()
        for i in range(0,self.n_layers):
            l = nn.Linear(input_size,input_size).to(self.factory_kwargs['device']).type(self.factory_kwargs['dtype'])
            a = nn.ReLU().to(self.factory_kwargs['device']).type(self.factory_kwargs['dtype'])
            encode_layers["enc_lin_"+str(i)] = l
            encode_layers["enc_act_" + str(i)] = a

        self.deep_body = nn.Sequential(encode_layers)
        decode_layers = OrderedDict()

        for i in range(0, self.n_layers):
            l = nn.Linear(input_size, input_size).to(self.factory_kwargs['device']).type(self.factory_kwargs['dtype'])
            a = nn.ReLU().to(self.factory_kwargs['device']).type(self.factory_kwargs['dtype'])
            decode_layers["dec_lin_"+str(i)] = l
            decode_layers["ded_act_" + str(i)] = a

        self.head = nn.Sequential(decode_layers)

        self.setup()


class RandomEncoder(DeepNet):

    def __init__(self,input_size=64,min_layers = 2, max_layers = 5, min_bottle=3, max_bottle = 10,
                 symmetrical=False,batch_norm = False, dropout = False):
        """

        :param input_size:
        :param min_layers: sample n layers from this range
        :param max_layers:
        :param min_bottle: sample bottleneck size from this range
        :param max_bottle:
        :param symmetrical:
        """
        super(RandomEncoder, self).__init__()
        self.input_size = input_size

        self.min_layers = min_layers
        self.max_layers = max_layers

        self.min_bottle = min_bottle
        self.max_bottle = max_bottle

        self.symmetrical = symmetrical

        self.act_funs = [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Hardtanh]
        self.batch_norm = batch_norm
        self.dropout = dropout

        #sample model size
        self.bottle_size = random.randint(min_bottle,max_bottle)
        self.n_layers = random.randint(min_layers,max_layers)
        #encode
        encode_layers = OrderedDict()
        previous_size = self.input_size
        for i in range(0,self.n_layers):
            sampled_size = random.randint(self.bottle_size,self.input_size)
            a = random.sample(self.act_funs,1)[0]()
            encode_layers["enc_"+str(i)] = self.init_layer(previous_size,sampled_size,act_function=a)
            previous_size = sampled_size
        self.encode = nn.Sequential(encode_layers)
        #decode
        decode_layers = OrderedDict()
        previous_size = self.bottle_size
        for i in range(0,self.n_layers):
            sampled_size = random.randint(self.bottle_size,self.input_size)
            a = random.sample(self.act_funs,1)[0]()
            decode_layers["dec_"+str(i)] = self.init_layer(previous_size,sampled_size,act_function=a)
            previous_size = sampled_size
        decode_layers["dec_lin_last"] = nn.Linear(previous_size,input_size,bias=False)
        self.decode = nn.Sequential(decode_layers)
        self.setup()

class ShallowEncoder(DeepNet):
    def __init__(self, input_size=64, bottle_size=64, activation=nn.Identity, bias=True):
        super(ShallowEncoder, self).__init__()

        self.input_size=input_size
        self.bottle_size=bottle_size
        self.deep_body = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=bottle_size, bias=bias),
            activation(),
        )
        self.head = nn.Sequential(
            nn.Linear(in_features=bottle_size, out_features=input_size, bias=bias),
            activation(),
        )
        self.setup()

class LinearReduceEncoder(DeepNet):
    def __init__(self,input_size=64,bottle_size=64,jump=1,activation=nn.Identity,bias=True):
        super(LinearReduceEncoder, self).__init__()
        self.input_size=input_size
        self.bottle_size=bottle_size

        self.encode= nn.Sequential()
        self.encode= nn.Sequential()

        #todo iteraly add layers to model
        self.setup()




if __name__ == "__main__":
    if False:
        parser = argparse.ArgumentParser(description="Run an experiment")
        parser.add_argument("data_name", action="store", default="", type=str, help="Input file name")
        parser.add_argument("output_name", action="store",help = "File to output to")
        parser.add_argument("--train_pct",dest="train_pct",action="store",type=float, help = "Percent to take as train split",)
        parser.add_argument("--bs",action="store", type=int, help="Batch size")
        parser.add_argument("--lr",action="store", type=float, help="Learning Rate")
        parser.add_argument("--epochs",action="store", type=int, help="Number of Epochs")
        parser.add_argument("--model",action="store", help = "Model Instance")
        parser.add_argument("--loss_fun",action="store", help = "Loss Function to use")
        parser.add_argument("--opt_fun",action="store", help = "Optimiser Function to use")
        parser.add_argument("--show_output",action="store", type = bool, help = "Flag to print full configurations or just outputs")

        pargs = parser.parse_args()

        loss_fun = eval(pargs.loss_fun)
        opt_fun = eval(pargs.opt_fun)
        model_name = pargs.model

    a_al_rt_file = 'D:/workspace/lazydeep/data/soil_data/A_AL_RT.csv'

    data = pd.read_csv(a_al_rt_file)
    nrow, ncol = data.shape
    n_features = ncol - 1

    # preprocessing
    from sklearn.preprocessing import StandardScaler

    class_name = "A_AL_RT"

    class_col = data[class_name]
    data = data.drop(columns=class_name)

    scaler = StandardScaler()
    scaler.fit(data)
    trans_mean = scaler.mean_
    trans_var = scaler.var_
    data = pd.DataFrame(scaler.transform(data))
    data[class_name] = class_col

    identity = ShallowEncoder(input_size=n_features, bottle_size=n_features, activation=nn.Identity)


    models = {'identity': identity}
    bs = 32
    lr = 1e-2
    epochs = 10
    opt_fun = optim.SGD
    loss_fun = nn.MSELoss
    lr_update = 'none'

    exp = experiment.Configuration(opt=opt_fun, loss=loss_fun, epochs=epochs, bs=bs, lr=lr)

    train_results = cross_evaluate(input=data, models=models, exp=exp, n_splits=10)



#for each architecture
    # learning rate
    # epochs
    # batch_size

#record batch and average loss
#visualise these