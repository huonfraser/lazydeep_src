from collections import OrderedDict

import torch
from sklearn.neighbors import KNeighborsRegressor
from torch import nn
from torch import optim
import random
from math import sqrt
from copy import deepcopy
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from torch.nn.parameter import Parameter
import tabular_deep
from sk_models import LocalWeightedRegression

from utils import *

class LWRHead(nn.Module):

    def __init__(self, device=None, dtype=None,n_neighbours=500):
        super(LWRHead,self).__init__()
        self.device=device
        self.dtype=dtype
        self.lwr = LocalWeightedRegression(n_neighbours=n_neighbours)

        #self.head = nn.Linear(num_inputs,num_outputs).to(device).type(dtype)
        #self.act = act_fun().to(device).type(dtype)

    def forward(self,X):
        X = X.detach().numpy()
        preds = self.lwr.predict(X)
        return torch.tensor(preds).to(self.device).type(self.dtype)

    def pretrain(self,X,y):
        """
        Fit our pls layers based on the data given
        :param X:
        :param y:
        :return:
        """
        self.lwr.fit(X,y)

class Head(nn.Module):

    def __init__(self,num_inputs,num_outputs,act_fun = nn.Identity, pretrain_head = False, device=None, dtype=None):
        super(Head,self).__init__()
        self.num_inputs=num_inputs
        self.num_outputs=num_outputs
        self.act_fun = act_fun
        self.device=device
        self.dtype=dtype
        self.pretrain_head = pretrain_head

        self.head = nn.Linear(num_inputs,num_outputs).to(device).type(dtype)
        self.act = act_fun().to(device).type(dtype)

    def forward(self,X):
        return self.act(self.head(X))

    def pretrain(self,X,y):
        """
        Fit our pls layers based on the data given
        :param X:
        :param y:
        :return:
        """
        if self.pretrain_head:
            #X1 = torch.Tensor(X)
            lin_reg = LinearRegression()
            lin_reg.fit(X.detach().numpy(),y)
            self.head.weight = Parameter(torch.Tensor(lin_reg.coef_.reshape(1,len(lin_reg.coef_))).to(self.device)).type(
                self.dtype)
            self.head.bias = Parameter(torch.Tensor([lin_reg.intercept_]).to(self.device)).type(
                self.dtype)


class DeepNet(nn.Module):
    deep_body: nn.Sequential
    head: Head
    reset_weights: nn.Parameter
    n_features: int
    n_layers : int

    #todo enforece n_features and n_layers
    
    def state(self):
        return self.state_dict()

    def load_state(self,file):
        self.load_state_dict(torch.load(file))
        return self
    
    def __init__(self, device=None, dtype=None):
        super(DeepNet, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}

    def reset(self):
        # self.load_state_dict(deepcopy(self.reset_weights))
        self.head.load_state_dict(self.head_reset_weights)
        self.deep_body.load_state_dict(self.body_reset_weights)
        return self

    def setup(self):
        #state_dict contains references, so we need to deepcopy
        #do we also need to detach?
        self.reseed()
        self.head_reset_weights = deepcopy(self.head.state_dict())
        self.body_reset_weights = deepcopy(self.deep_body.state_dict())

    def reseed(self):
        """
        Apply our initial weightts
        used to give cv runs all the same start point
        :return:
        """
        self.deep_body.apply(self.init_weights)
        self.head.apply(self.init_weights)

    def compress(self, x):
        """
        Extract features
        :param x:
        :return:
        """
        if self.deep_body is None:
            return x
        else:
            return self.deep_body(x)

    def forward(self, X, *args, **kargs):
        """
        make predictions
        :param x:
        :param args:
        :param kargs:
        :return:
        """
        X1 = self.compress(X)
        #print(f"DeepNet forward shape is {X1.shape}")
        if self.head is None:
            return X1
        else:
            #print(f"Head: {self.head.weight}")
            pred = self.head(X1)
            return pred.reshape(len(pred),)

    def init_weights(self, m):
        """
        Function to initialse weighs for layers, to e used with apply
        :param m:
        :return:
        """
        if type(m) == nn.Linear:
            stdv = 1. / sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

    def init_layer(self, in_size=50, out_size=50, dropout=0.001,
                   act_function=nn.ReLU):
        layer = OrderedDict()
        l = nn.Linear(in_size, out_size).to(self.factory_kwargs['device'])#.type(self.factory_kwargs['dtype'])

        layer["linear"] = l
        if self.batch_norm:
            layer["batch_norm"] = nn.BatchNorm1d(out_size).to(self.factory_kwargs['device'])#.type(self.factory_kwargs['dtype'])
        layer["act_fun"] = act_function.to(self.factory_kwargs['device'])#.type(self.factory_kwargs['dtype'])
        if self.dropout:
            layer["dropout"] = nn.Dropout(dropout).to(self.factory_kwargs['device'])#.type(self.factory_kwargs['dtype'])
        return nn.Sequential(layer)

class ApproxLinearRegression(DeepNet):

    def __init__(self,n_features=64,device=None, dtype=None):
        super(ApproxLinearRegression, self).__init__(device=device, dtype=dtype)
        self.deep_body= None
        self.head = nn.Sequential(nn.Linear(n_features,1))
        self.n_features=65
        self.n_layers = 0

class RandomNet(DeepNet):

    def __init__(self, input_size=64, n_features=None, min_layers=2, max_layers=10, n_layers=None,
                 act_function=None, allow_increase_size=False, dropout=True, batch_norm=False, device=None, dtype=None):
        """
        Either sample a network depth parameter or take a given option, then build a random network
        :param input_size:
        :param min_layers:
        :param max_layers:
        """

        super(RandomNet, self).__init__(device=device,dtype=dtype)
        self.input_size = input_size

        self.min_layers = min_layers
        self.max_layers = max_layers

        self.dropout = dropout
        self.batch_norm = batch_norm

        if n_features is None:
            self.n_features = self.input_size
        else:
            self.n_features = n_features

        #set up choice of activation functions
        if act_function is None:
            self.act_funs = [nn.ReLU]
        else:
            self.act_funs = [act_function]

        # sample model size
        if n_layers is None:
            self.n_layers = random.randint(min_layers, max_layers)
        else:
            self.n_layers = n_layers

        if allow_increase_size: #case where every layer is randomized
            layers = OrderedDict()
            previous_size = self.input_size
            for i in range(0, self.n_layers):
                a = random.sample(self.act_funs, 1)[0]()
                d_out = (random.uniform(0.01, 0.1))
                sampled_size = random.randint(1, self.n_features)
                layers[f"block_{i}"] = self.init_layer(previous_size,sampled_size,act_function=a,dropout=d_out)
                previous_size = sampled_size
            self.deep_body = nn.Sequential(layers)
            self.head = nn.Sequential(nn.Linear(previous_size, 1,**self.factory_kwargs))

        else: #case where our network shrinks
            #start with last_layer
            last_layer_size = self.n_features
            self.head = nn.Sequential(nn.Linear(last_layer_size, 1))

            layers = OrderedDict()
            previous_size = self.input_size
            for i in range(0, self.n_layers):
                if i == self.n_layers -1: # make sure we match the last layers
                    sampled_size = last_layer_size
                else:
                    sampled_size = random.randint(last_layer_size, previous_size)
                d_out = (random.uniform(0.01, 0.1))
                a = random.sample(self.act_funs, 1)[0]()
                layers[f"block_{i}"] = self.init_layer(previous_size,sampled_size,act_function=a,dropout=d_out)
                previous_size = sampled_size
            self.deep_body = nn.Sequential(layers)
        self.setup()



class CustomDeep(DeepNet):
    def __init__(self, deep_body:nn.Sequential=None, head:Head=None,n_features=0,*args,**kwargs):
        super(CustomDeep, self).__init__(*args,**kwargs)
        self.deep_body = deep_body
        self.head = head
        self.n_features=n_features

        self.setup()


class CNNBaseline(DeepNet):
    
    def __init__(self, d_in=50, n_blocks=3, d_main=50, d_hidden=50, dropout_first=True,
                                          dropout_second=True,d_out=50):
        super(CNNBaseline, self).__init__()
        self.resnet = tabular_deep.ResNet(d_in = d_in,
        n_blocks=n_blocks,
        d_main=d_main,
        d_hidden=d_hidden,
        dropout_first=dropout_first,
        dropout_second=dropout_second,
        normalization='BatchNorm1d',
        activation='ReLU',
        d_out=d_out)

    def forward(self, X):
        return self.resnet.forward(X)

    def compress(self, X):
        return self.resnet.head.forward(X)

        
class TransformerBaseline(DeepNet):
    def __init__(self):
        super(TransformerBaseline, self).__init__()
        from ft_transformer import Transformer
        self.transformer = Transformer(
        d_numerical= int,
        categories= None,
        token_bias= None,
        n_layers= int,
        d_token= int,
        n_heads= int,
        d_ffn_factor= float,
        attention_dropout= float,
        ffn_dropout=float,
        residual_dropout= float,
        activation= str,
        prenormalization=bool,
        initialization= str,
        kv_compression= None,
        kv_compression_sharing= None,
        d_out= int
        )

    def compress(self, X):
        return self.transformer.compress(X)

    def forward(self, X):
        return self.transformer.forward(X)

def load_torches(dir:str, names:list):
    """
    Helper function to load a dict of models
    :param dir:
    :param names:
    :return:
    """
    return {name: torch.load(dir+'/'+name) for name in names}



class RandomPLSNet(RandomNet):
    """
    We specify number of features to be <= our max pls_size

    #todo, should we add an activation function after the first layer, maybe we should init our model as ones
    """
    def __init__(self, max_pls_size = 50, fix_pls = False,
                 input_size=64, n_features = 64, min_layers=2, max_layers=10, n_layers=None,
                 act_function = None, allow_increase_size = False, dropout=True, batch_norm = False):

        self.pre_pls_size = input_size
        self.pls_features = random.uniform(n_features,max_pls_size)
        self.fix_pls = fix_pls
        self.first_layer = None
        self.pls = PLSRegression(n_components=self.pls_features)

        super().__init__(input_size=self.pls_features, n_features =n_features, min_layers=min_layers,
                         max_layers=max_layers, n_layers=n_layers, act_function = act_function,
                         allow_increase_size = allow_increase_size, dropout=dropout, batch_norm = batch_norm)

    def train_pls(self,X,y):
        self.pls.fit(X,y)

        if self.fix_pls == False: #create a layer
            self.first_layer = nn.Linear(self.pre_pls_size,self.pls_features,bias=False)
            weights = self.pls.x_rotations_
            with torch.no_grad():
                self.first_layer.weight = Parameter(weights)



    def forward(self, x, *args, **kargs):
        if self.fix_pls == False: #our pls is just a normal layer
            pred = self.last_layer(self.layers(self.first_layer(x)))
        else:
            #todo, back to np
            x = self.pls.transform(x)
            #todo, back to tensor
            pred = self.last_layer(self.layers(x))
            pred = -1
        return pred.reshape(len(pred), )




class DeepKNN(DeepNet):

    def __init__(self, deep_net, n_neighbours=5, type="regression"):
        super(DeepKNN, self).__init__()

        self.layers = deep_net.layers

        self.knn = KNeighborsRegressor(n_neighbors=5)

    def forward(self, x, *args, **kargs):
        features = self.layers(x)

        return self.knn(features.squeeze().numpy())


class DualKNN(DeepNet):

    def __init__(self, deep_net, n_neighbours=5, type="regression"):
        super(DualKNN, self).__init__()

        self.layers = deep_net.layers
        self.last_layer = deep_net.last_layer

        self.knn = KNeighborsRegressor(n_neighbors=5)

    def forward(self, x, *args, **kargs):
        features = self.layers(x)

        knn_result = self.knn(features.squeeze().numpy())
        deep_result = self.knn()

        # cd = (knn_result^0.5)*(deep_result^0.5) #multiplicative loss

        return 0.5 * knn_result + 0.5 * deep_result  # linear loss
