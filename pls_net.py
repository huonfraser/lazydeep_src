from collections import OrderedDict
from copy import deepcopy
from random import random

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from torch import Tensor, matmul, optim
from torch.nn import Module, Linear, Parameter, Sequential, Identity, Dropout, ReLU, MSELoss
import torch
import numpy as np
import pandas as pd

from configurations import Configuration
from deep_net import DeepNet, Head, CustomDeep, LWRHead
from evaluation import DeepScheme
from utils import TabularDataset


class LWRLayer(Module):
    """
    Class that wraps a locally weighted regression into a pytorch layer
    This needs to calculate the gradients and if its not possible then i need to implement as a loss function instead
    """
    def __init__(self, n_neighbours=100,device=None,dtype=None):
        super(LWRLayer,self).__init__()
        self.n_neighbours=n_neighbours
        self.device=None,
        self.dtype=None
        self.linear_approx = None #

    def forward(self,X):
        #for each
        #look up from db -assume db is tensors
        for row in torch.unbind(X):
            # calculate distance  between points
            dist = torch.nn.MSELoss(row,self.db,reduction=None)
            topk =torch.topk(self.n_neighbours, largest=False)
            indices = topk.indices
            # take best n

            #train regresion
            #predict
            pass





        pass

    @staticmethod
    def backward(self,grad_output):
        #best bet is to build a linear regression model and use that as our approximation for backwards pass
        pass

class PLSLayer(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    def __init__(self,in_features= None,out_features = None, device=None, dtype=None, fixed=False):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super(PLSLayer,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pls = PLSRegression(n_components=out_features,scale=True)
        self.fixed = fixed
        self.linear = None #Linear(in_features,out_features,bias=False)

        for param in self.parameters():
            param.requires_grad = False

    def fit(self, X, y):
        self.pls.fit(X,y)
        weights = np.transpose(self.pls.x_rotations_)
        self.reset_weights = weights
        self.x_mean = Tensor(self.pls.x_mean_)
        self.x_std = Tensor(self.pls.x_std_)
        #print(f"PLS Weights {self.reset_weights}")
        if not self.fixed:
            self.set_linear(weights)

        #self.register_parameter(name='linear',param=self.linear)

    def forward(self, X):
        if self.fixed:
            X1 = X.numpy()
            X2 = self.pls.transform(X1)
            X3 = torch.tensor(X2).to(self.factory_kwargs['device']).type(self.factory_kwargs['dtype'])
            return X3
        else:


            X1 = (X - self.x_mean)/self.x_std
            return self.linear(X1)


    #@staticmethod
    #def backward(gradient=None, retain_graph=None, create_graph=False, inputs=None):
    #   print("backs")
    #   if not self.fixed:
    #        output = self.linear.backward(gradient)  # here you call the function!
    #
     #       return output
    #   else:
    #       return None

    def reset(self):
        #self.linear = Linear(self.in_features,self.out_features,bias=False)
        #weights = np.transpose(self.pls.x_rotations_)
        #self.set_linear(weights)
        pass

    def toggle(self):

        self.fixed = not self.fixed
        if self.fixed: # if now fixed (vaiiable -> fixed)
            self.linear = None
        else: #fixed -> variable
            self.set_linear(self.reset_weights)
           # self.register_parameter(name='linear',param=self.linear)

        #self.set_linear((self.reset_weights))

    def set_linear(self,weights):
        self.reset_weights = weights
        #self.linear = Linear(self.in_features, self.out_features, bias=False)
        #new_weights = Parameter(Tensor(weights).float())#to(self.factory_kwargs['device']).type(self.factory_kwargs['dtype'])
        #print(f"New weights: {new_weights}")
        #self.linear.weight = new_weights
        #print(f"Set weights as {self.linear.weight}")
        self.linear = Linear(self.in_features,self.out_features,bias=False)
        self.linear.weight = Parameter(Tensor(weights).float())


class PLSNet(DeepNet):
    pls_body : Sequential
    def __init__(self,
                 n_features=64,
                 n_components =64,
                 n_layers = 0,
                 deep_body = None,
                 n_outputs = 64,
                 fix_pls = True,
                 pretrain_head = False,
                 head_type = None, *args, **kwargs):
        super(PLSNet, self).__init__(*args, **kwargs)
        self.n_features = n_features
        self.n_components = n_components
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.fix_pls = fix_pls
        self.head_type = head_type
        self.pretrain_head = pretrain_head
        self.pls_body = PLSLayer(n_features,n_components,fixed= fix_pls,*args,**kwargs)

        #setup deep_body
        if deep_body is not None:
            self.deep_body = deep_body
            self.n_outputs = self.n_components
        elif n_layers <= 0:
            self.deep_body=Sequential(Identity())
            self.n_outputs = self.n_components
        else:
            layers = OrderedDict()
            previous_size = self.n_components
            for i in range(0, self.n_intermediate_layers):
                sampled_size = random.randint(1, self.n_features)
                d_out = (random.uniform(0.01, 0.1))
                a = random.sample(self.act_funs, 1)[0]()
                layers["lin_" + str(i)] = self.init_layer(previous_size, sampled_size, act_function=a, dropout=d_out)
                previous_size = sampled_size
            self.deep_body = Sequential(layers)
        #setup head
        if self.head_type is None:
            #self.head = Linear(self.n_outputs,1).to(self.factory_kwargs['device']).type(self.factory_kwargs['dtype'])
            self.head = Head(num_inputs=self.n_outputs,num_outputs=1,act_fun=Identity,device=self.factory_kwargs['device'],dtype=self.factory_kwargs['dtype'],pretrain_head=pretrain_head)
        elif self.head_type == "lwr":
            self.head = LWRHead(device=self.factory_kwargs['device'],dtype=self.factory_kwargs['dtype'])
        self.setup()

    def pretrain(self, data):
        X,y = zip(*[(X,y) for X,y in data])
        self.pls_body.fit(X,y)
        X = self.pls_body(torch.Tensor(X).float())
        X = self.deep_body(X)
        #todo prefit body
        self.head.pretrain(X,y)

    def prefit(self,X,y):
        """
        Fit our pls layers based on the data given
        :param X:
        :param y:
        :return:
        """
        self.pls_body.fit(X,y)
        X = self.pls_body(torch.Tensor(X).float())
        X = self.deep_body(X)
        #todo prefit body
        self.head.pretrain(X,y)


    def compress(self, X):
        X = self.pls_body(X)
        if self.deep_body is None:
            return X
        else:
            return self.deep_body(X)


    def reset(self):
        super(PLSNet, self).reset()
        self.pls_body.reset()
        return self


class SimplePLSNet(PLSNet):

    def __init__(self, *args, **kwargs):
        super(SimplePLSNet, self).__init__(*args, **kwargs)

class RandomPLSNet():

    def __init__(self):
        pass


class GreedyPLSNet():
    """Greedily build a model

    Here we are going variable initialisaiation
    """
    def __init__(self,max_components=100,pls_fixed= True,head_fixed = True,act_fun = ReLU,train=False,device="cpu",dtype=torch.float
                 ,head_type=None):
        self.max_components=max_components
        self.head_fixed = head_fixed
        self.pls_fixed = pls_fixed
        self.act_fun = act_fun
        self.train=train
        self.fixed_hyperparams = {'bs': 32,'loss': MSELoss(),'epochs': 100}#todo add params for these
        self.device=device
        self.dtype=dtype
        self.head_type=head_type
        self.config = Configuration(
        opt = optim.SGD,
        lr = 1e-5,
        lr_update= torch.optim.lr_scheduler.ReduceLROnPlateau,#torch.optim.lr_scheduler.ReduceLROnPlateau,
        n_layers = 0,
        act_function= self.act_fun,
        allow_increase_size = False,
        n_features = None,
        dropout = False,
        batch_norm = False)
        #hyperparams  are
        #1) train head - prefit or deep train
        #2) pls bodys variable or not on train
        #3) make pls bodys variable for final model and do we further train this model


        #finally - can we do end to end


    def build(self,train_data,val_data):

        #todo, start with fixed, then add variable, then add both

        dict = OrderedDict()
        num_inputs = train_data.num_features
        num_features = self.max_components

        #setup step 1

        models_step1 = {i: PLSNet(n_features=num_inputs,
                 n_components =i,pretrain_head=self.head_fixed,fix_pls=self.pls_fixed,
                                  device=self.device,dtype=self.dtype,
                                  head_type=self.head_type) for i in range(2,num_features)}
        configs_step1 = {i : deepcopy(self.config) for i in range(2,num_features)}
        scheme_step1 = DeepScheme(models_step1, configs_step1,fixed_hyperparams=self.fixed_hyperparams,logger=None)

        #run step 1
        scheme_step1.pretrain(train_data)
        if self.train:
            models_post_step1, losses_step1 = scheme_step1.train(train_data,val_data)
        scores_step1, preds_step1 = scheme_step1.test(val_data)
        best_model_step1 = min(scores_step1, key=scores_step1.get)
        #find best result
        best_score_step1 = scores_step1[best_model_step1]
        dict["pls_1"] = models_step1[best_model_step1].pls_body
        dict["act_1"] = self.act_fun()
        model_so_far = Sequential(dict)
        print(f"Build layer one, with {best_model_step1} components, giving a loss of {best_score_step1}")

        #transform our data
        X, y = zip(*[(X, y) for X, y in train_data])
        X = model_so_far(torch.tensor(X).float()).detach().numpy()

        intermediate_data = pd.DataFrame(X)
        intermediate_data['y'] = y
        intermediate_train = TabularDataset(intermediate_data,output_cols=['y'])

        X, y = zip(*[(X, y) for X, y in val_data])
        X = model_so_far(torch.tensor(X).float()).detach().numpy()

        intermediate_data = pd.DataFrame(X)
        intermediate_data['y'] = y
        intermediate_val = TabularDataset(intermediate_data, output_cols=['y'])
        num_features = best_model_step1

        #setup step 2
        models_step2 = {i: PLSNet(n_features=num_features,
                 n_components =i,pretrain_head=self.head_fixed,fix_pls=self.pls_fixed,
                                  device=self.device,dtype=self.dtype,
                                  head_type=self.head_type) for i in range(2,num_features+1)}
        configs_step2 = {i : deepcopy(self.config) for i in range(2,num_features+1)}
        scheme_step2 = DeepScheme(models_step2, configs_step2,fixed_hyperparams=self.fixed_hyperparams,logger=None)

        #run step 2
        scheme_step2.pretrain(intermediate_train)
        if self.train:
            models_post_step2, losses_step2 = scheme_step2.train(intermediate_train,intermediate_val) #todo val data
        scores_step2, preds_step2 = scheme_step2.test(intermediate_val)
        #find best result
        best_model_step2 = min(scores_step2, key=scores_step2.get)
        best_score_step2 = scores_step2[best_model_step2]

        dict["pls_2"] = models_step2[best_model_step2].pls_body
        #dict["act_1"] = self.act_fun()
        model_so_far = Sequential(dict)
        print(f"Build layer two, with {best_model_step2} components, giving a loss of {best_score_step2}")
        head = models_step2[best_model_step2].head

        return CustomDeep(deep_body = model_so_far,head=head)

        #step one - search over models with n_components
        #either then pass into linear regression - or go based on fit

        #then -for different activation functions (start with just sigmoid)
            #pass through so far, then build another layer
        self.deep_body= Sequential(dict)
        #pass through X
        self.head = Head()
        self.head.prefit(X,y)
        pass