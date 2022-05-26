import jsonpickle
import numpy as np
import torch
import random 
from sk_models import LinearRidge
from sklearn.linear_model import LinearRegression
from pipeline import StreamLearner
import collections
import math

from river.neighbors import KNNRegressor
from river.utils import numpy2dict
from river.ensemble import SRPRegressor,BaggingRegressor
from river import base
from tqdm.notebook import tqdm, trange
from river.metrics import RegressionMetric
from sklearn.metrics import mean_squared_error,r2_score
import evaluation as ev
import statistics
import scipy
import pandas as pd
class ExponentialHistogram():

    
    def __init__(self,max_width=10000):
        self.bucket_deque: Deque['Bucket'] = deque([Bucket()])
        self.width = 0.
        self.n_buckets = 0
        self.max_n_buckets = 0
        self.min_window_length = 5
        self.max_width = 10000
        
        
    def reset(self):
        self.__init__()
        
    def update(self,value:float):
        """
        Update the change detector with a single data point.
        Apart from adding the element value to the window, by inserting it in
        the correct bucket,
        
        we will not be - it will also update the relevant statistics, in
        this case the total sum of all values, the window width and the total
        variance.
        """
        self._insert_element(value, 0.0)
        
        
    def _insert_element(self, value, variance):
        bucket = self.bucket_deque[0]
        bucket.insert_data(value, variance)
        self.n_buckets += 1

        if self.n_buckets > self.max_n_buckets:
            self.max_n_buckets = self.n_buckets

        # Update width, variance and total
        self.width += 1

        self._compress_buckets()    

    def _calculate_bucket_size(row: int):
        return pow(2, row)
        
    def _delete_element(self):
        bucket = self.bucket_deque[-1]
        n = self._calculate_bucket_size(len(self.bucket_deque) - 1) # length of bucket

        # Update width, total and variance
        self.width -= n

        bucket.remove()
        self.n_buckets -= 1

        if bucket.current_idx == 0:
            self.bucket_deque.pop()

        return n    
    
    def values(self):
        values = [] 
        
        for bucket in self.bucket_deque():
            s = bucket.current_idx
            for i in range(0,s):
                values.append(bucket.ge_total_at(s))
                
        return values
    
    def  _compress_buckets(self):
        
        bucket = self.bucket_deque[0]
        idx = 0
        while bucket is not None:
            k = bucket.current_idx
            # Merge buckets if there are more than MAX_BUCKETS
            if k == self.MAX_BUCKETS + 1:
                try:
                    next_bucket = self.bucket_deque[idx + 1]
                except IndexError:
                    self.bucket_deque.append(Bucket())
                    next_bucket = self.bucket_deque[-1]
                n1 = self._calculate_bucket_size(idx)   # length of bucket 1
                n2 = self._calculate_bucket_size(idx)   # length of bucket 2
                mu1 = bucket.get_total_at(0) / n1       # mean of bucket 1
                mu2 = bucket.get_total_at(1) / n2       # mean of bucket 2

                # Combine total and variance of adjacent buckets
                if random.random() < 0.5:
                    total12 = bucket.get_total_at(0)
                else:
                    total12 = bucket.get_total_at(1)
                next_bucket.insert_data(total12, 0)
                self.n_buckets += 1
                bucket.compress(2)

                if next_bucket.current_idx <= self.MAX_BUCKETS:
                    break
            else:
                break

            try:
                bucket = self.bucket_deque[idx + 1]
            except IndexError:
                bucket = None
            idx += 1    
    

    
    
    
class ExponentialKNeighborsBuffer():
    
    """we need to store data in (x,y) form """
    
    
    def __configure(self):
        self.data = ExponentialHistogram()
        self._is_initialized = True
    
    
    def reset(self):
        self._n_features = -1
        self._n_targets = -1
        self._size = 0
        self._next_insert = 0
        self._oldest = 0
        self._imask = None
        self._X = None
        self._y = None
        self.data = None
        self._is_initialized = False
        
        
    def append(self,x,y):
        if not self._is_initialized:
            self._n_features = get_dimensions(x)[1]
            self._n_targets = get_dimensions(y)[1]
            self._configure()

        if self._n_features != get_dimensions(x)[1]:
            raise ValueError(
                "Inconsistent number of features in X: {}, previously observed {}.".format(
                    get_dimensions(x)[1], self._n_features
                )
            )
            
        self.data.update((x,y))
        self.size=self.data.width
        
        return self
        
    def clear(self):
        self._size = 0
        self.data = ExponentialHistogram()
        
        return self
    
    def features_buffer(self):
        x,_ = self.data.values()
        return x
    
    def targets_buffer(self):
        _,y = self.data.values()
        return y
        

        

class StreamLocalWeightedRegression(KNNRegressor):

    def __init__(self,
            n_neighbors = 5,
            window_size = 10000,
            leaf_size= 30,
            p= 2,
            ridge = False,
            floor = False,
            kernal = False,
            distance_kernal = 'uniform',
            convolution = 'additive',
            error_kernal='triangle',
            parametric = True,
            reverse = False,
            ridge_regression_param = 0.01,**kwargs):
        super().__init__(
            n_neighbors=n_neighbors,
            window_size=window_size,
            leaf_size=leaf_size,
            p=p,
            **kwargs
        )
        self.floor= floor
        self.ridge = ridge
        self.ridge_regression_param = ridge_regression_param
        self.kernal = kernal
        self.distance_kernal = distance_kernal
        self.error_kernal = error_kernal
        self.convolution = convolution
        self.parametric = parametric
        self.reverse=reverse
        #self.n_neighbours = 5
        
    def learn_one(self,x,y):
        x_arr = dict2numpyordered(x)
        self.data_window.append(x_arr, y)
        return self

    def predict_one(self,x,errors=None):
        if self.data_window.size == 0:
            # Not enough information available, return default prediction
            return 0.0
        
        x_arr = dict2numpyordered(x)
        dists, neighbor_idx = self._get_neighbors(x_arr)
        y_buffer = self.data_window.targets_buffer
        X_buffer = self.data_window.features_buffer
        

        #setlect our weights 
        if self.data_window.size < self.n_neighbors:  # Select only the valid neighbors
            neighbor_X = [
                X_buffer[index]
                for cnt, index in enumerate(neighbor_idx[0])
                if cnt < self.data_window.size
            ]

            neighbor_y = [
                y_buffer[index]
                for cnt, index in enumerate(neighbor_idx[0])
                if cnt < self.data_window.size
            ]
            if not errors is None:
                neighbor_errors = [
                    errors[index]
                    for cnt, index in enumerate(neighbor_idx[0])
                    if cnt < self.data_window.size
                ]
            dists = [
                dist for cnt, dist in enumerate(dists[0]) if cnt < self.data_window.size
            ]
        else:
            neighbor_X = [X_buffer[index] for index in neighbor_idx[0]]
            neighbor_y = [y_buffer[index] for index in neighbor_idx[0]]
            dists = dists[0]
            if not errors is None:
                neighbor_errors = [errors[index] for index in neighbor_idx[0]]
        
        #consider our four cases
        if errors is None: #uniform error_kernal
            if self.distance_kernal == 'uniform':
                weights =  np.asarray([1 for i in range(len(neighbor_y))])
            else: #triangle distance_kernal (weights should sum to one)
                if np.max(dists)>0:
                    weights = dists/np.max(dists) 
                else:   #distances are now scaled to be in the range (0,1) #apply_kernal 
                    weights = np.asarray([1 for i in dists])
        else: #triangle error_kernal
            if max(neighbor_errors)<=0:
                neighbor_errors = np.asarray([1 for i in range(0,len(neighbor_errors))])
            else:
                neighbor_errors = neighbor_errors/np.max(neighbor_errors) #errors are now scaled to be in the range (0,1)
                            #apply_kernal 
                if self.reverse:
                    neighbor_errors = 1-neighbor_errors
                    
            if self.distance_kernal == 'uniform':
                weights = neighbor_errors 
            else: #two triangles
                if np.max(dists)>0:
                    dists = dists/np.max(dists) 
                if self.convolution =='additive':
                    weights = (2-neighbor_errors-dists)
                else:
                    weights = (1-neighbor_errors)*(1-dists)
                    
                
        #check weights are validW            
        sum_weights = np.sum(weights)
        if sum_weights <= 0:
            weights = np.asarray([1 for i in weights])
            sum_weights = np.sum(weights)
            
        if self.parametric: #build weighted linear                    
            model = LinearRegression()
            model.fit(np.asarray(neighbor_X),neighbor_y,weights/sum_weights)
            pred = model.predict(np.asarray([x_arr]).reshape(1, -1))[0]
        else: #vote 
            votes = [weights[i]*neighbor_y[i] for i,_ in enumerate(weights)]
            pred= np.sum(votes)/sum_weights
        
        if self.kernal: #restrict predictions
            top_y = max(neighbor_y)
            bot_y = min(neighbor_y)
            if pred > top_y:
                pred = top_y
            elif pred < bot_y:
                pred = bot_y
              
        return pred



    def from_state(self,state):
        pass
    
    def state(self):
        state_dict = {'ridge':self.ridge,
                     'ridge_regression_param':self.ridge_regression_param,
                     'kneighbours_state':self.kneighbours.state()}
        
        return state_dict

    def reset(self):
        super().reset()

    
class ExpHistLocalWightedregression(StreamLocalWeightedRegression):
        
        def __init___(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.data_window = ExponentialKNeighborsBuffer()
            
           


class StreamDeep(base.SupervisedTransformer):
    
    def __init__(self,deep):
        self.deep=deep
        #._supervised = True
        
    def learn_one(self,x,y):
        return self
    
    def predict_one(self,x):
        x_arr = dict2numpyordered(x)
        x_arr1 = x_arr.reshape(1,len(x_arr))
        pred = self.deep.forward(torch.Tensor(x_arr1).float()).detach().numpy()

        return pred[0]
    
    def transform_one(self,x):
        x_arr = [dict2numpyordered(x)]
        x_arr = self.deep.compress(torch.tensor(x_arr).float()).detach().numpy()
        x = numpy2dict(x_arr[0])
        
        return x
    
    
class StreamWrapper(base.SupervisedTransformer):        
    def __init__(self,model):
        """
        Wrapper for a batch model, where we freeze in place. 
        
        """
        self.model=model
        #self._supervised = True
        
    
    def learn_one(self,x,y):
        return self
    
    def transform_one(self,x):
        x_arr = dict2numpyordered(x)
        x_arr1 = x_arr.reshape(1,len(x_arr))
        pred = self.model.transform(x_arr1)
        return(numpy2dict(pred[0]))
    
    def predict_one(self,x):
        x_arr = dict2numpyordered(x)
        x_arr1 = x_arr.reshape(1,len(x_arr))
        pred = self.model.predict(x_arr1)

        return pred[0]    
    
class StreamBoost(base.Regressor):
    def __init__(self,deep,ws=10000,n_neighbors=5,kernal=False,convolution='additive',parametric=True,reverse=False,distance_kernal='triangle'):
        self.deep=deep
        self.lwr=StreamLocalWeightedRegression(n_neighbors=n_neighbors,kernal=kernal,convolution=convolution,parametric=parametric,window_size=ws,reverse=reverse,distance_kernal=distance_kernal)
        self.ws=ws
        self.errors=deque()
        
    def learn_one(self,x,y):
        x_arr = [dict2numpyordered(x)]
        
        #calculate and store error
        pred = self.deep.forward(torch.tensor(x_arr).float()).detach().numpy()[0]
        e = np.abs(pred-y)
        if len(self.errors) == self.ws:
            self.errors.popleft()
        self.errors.append(e)
        
        #extract features and learn 
        x_arr = self.deep.compress(torch.tensor(x_arr).float()).detach().numpy()
        x = numpy2dict(x_arr[0])
        self.lwr.learn_one(x,y)
        

    
    def predict_one(self,x):
        x_arr = [dict2numpyordered(x)]
        x_arr = self.deep.compress(torch.tensor(x_arr).float()).detach().numpy()
        x = numpy2dict(x_arr[0])
        pred = self.lwr.predict_one(x,errors=self.errors)
        return pred
            
    
    
    
            
class StreamDeepLWR(base.Regressor):
    
    def __init__(self,scaler,deep,lwr):
        self.scaler=scaler
        self.deep=deep
        self.lwr=lwr
        
    def learn_one(self,x,y):
        x_arr = [dict2numpyordered(x)]
        x_arr = self.scaler.transform(x_arr)
        x_arr = self.deep.compress(torch.tensor(x_arr).float()).detach().numpy()
        x = numpy2dict(x_arr[0])
        self.lwr.learn_one(x,y)
    
    def predict_one(self,x):
        x_arr = [dict2numpyordered(x)]
        x_arr = self.scaler.transform(x_arr)
        x_arr = self.deep.compress(torch.tensor(x_arr).float()).detach().numpy()
        x = numpy2dict(x_arr[0])
        pred = self.lwr.predict_one(x)
        return pred
    
class StreamMixup(base.Transformer):
    def __init__(self, alpha=0.001):
        self.with_std = True
        self.counts = collections.Counter()
        self.means = collections.defaultdict(float)
        self.vars = collections.defaultdict(float)
        self.alpha=alpha
        
    def learn_one(self, x):
        for i, xi in x.items():
            self.counts[i] += 1
            old_mean = self.means[i]
            self.means[i] += (xi - old_mean) / self.counts[i]
            if self.with_std:
                self.vars[i] += (
                    (xi - old_mean) * (xi - self.means[i]) - self.vars[i]
                ) / self.counts[i]

        return self
    

    def transform_one(self, x):
        if self.counts[0] < 2:
            return x
        return {i: xi +self.alpha*np.random.normal(self.means[i],self.vars[i]) for i, xi in x.items()}
    
    def reset(self):
        self.counts = collections.Counter()
        self.means = collections.defaultdict(float)
        self.vars = collections.defaultdict(float)
    
    
class MixupEnsemble(SRPRegressor):    
    
    def __init__(self,alpha=0.01,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mixup = StreamMixup(alpha=alpha)
            
    def learn_one(self, x: dict, y, **kwargs):
        if not self:
            self._init_ensemble(list(x.keys()))
        
        self.mixup.learn_one(x)
        
        for model in self:
            # Get prediction for instance
            y_pred = model.predict_one(x)
            if y_pred is None:
                y_pred = 0

            # Update performance evaluator
            model.metric.update(y_true=y, y_pred=y_pred)

            # Train using random subspaces without resampling,
            # i.e. all instances are used for training.
            if self.training_method == 'mixup':
                k = 1.0
            elif self.training_method == self._TRAIN_RANDOM_SUBSPACES:
                k = 1.0
            # Train using random patches or resampling,
            # thus we simulate online bagging with Poisson(lambda=...)
            else:
                k = self._rng.poisson(lam=self.lam)
                if k == 0:
                    # skip sample
                    return self
            model.learn_one(
                x=self.mixup.transform_one(x),
                y=y,
                sample_weight=k,
                n_samples_seen=self._n_samples_seen,
                rng=self._rng,
            )
        

        return self
            
    def reset(self):
        super().reset()
        self.mixup.reset()
        self.data = []
        self._n_samples_seen = 0
        self._rng = np.random.default_rng(self.seed)

    
def dict2numpyordered(data_):
    return np.asarray(list(x for _, x in data_.items()))

def prequential_evaluate(dataset,models,metrics_,pretrain=10000,num_its=100000):
    """
    only make prediction if after pretrain length
    """
    X,y = dataset[0:num_its]
    
    preds = {name:[] for name in models.keys()}
    preds['y'] = []
    
    scores = {k:{name:[] for name in models.keys()} for k in metrics_.keys()}   
    
    for i in tqdm(range(0,num_its)):
        xi = numpy2dict(X[i])
        yi = y[i]
        
        
        preds['y'].append(yi)    
            
        for name,model in models.items():  
            pred = model.predict_one(xi)
            preds[name].append(pred)
            #predict if pretrained
            if i >= pretrain:
                for metric_k,metric_v in metrics_.items():
                    score = metric_v[name].update(yi, pred).get()
                    scores[metric_k][name].append(score)
                    
            #learn
            model.learn_one(xi,yi)


    return preds, scores,models,metrics_

def prequential_evaluate_torch(dataset,models,metrics_,opts=None,lrs=None,pretrain=10000,bs=32,pp=None):
    #todo, pretrain
    preds = {name:[] for name in models.keys()}
    preds['y'] = []
    loss_eval = ev.loss_target
    loss_fun =  torch.nn.MSELoss()
    
    scores = {k:{name:[] for name in models.keys()} for k in metrics_.keys()}   
    
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)
    for batch, (X, y) in tqdm(enumerate(train_loader)):
        if pp is None:
            X = X.float()
        else:
            X = torch.Tensor(pp.transform(X.detach().numpy())).float()
        y = y.float()
        test_preds, test_losses = ev.test_batch(X, y, models, opts, loss_eval,loss_fun,time=False)
        for yi in y:
            preds['y'].append(yi.detach().numpy())
        for k,v in test_preds.items():
            for i,j in enumerate(v):
                preds[k].append(j)
                for metric_k,metric_v in metrics_.items():
                    score = metric_v[k].update(y[i].detach().numpy(), j).get()
                    scores[metric_k][k].append(score)
        
        train_losses = ev.train_batch(X, y, models, opts, loss_eval, loss_fun,update = True)
        
    return preds, scores,models,metrics_

def score_evaluate_torch(dataset,models,metrics_,opts=None,lrs=None,bs=32,pp=None):
    #todo, pretrain
    preds = {name:[] for name in models.keys()}
    preds['y'] = []
    loss_eval = ev.loss_target
    loss_fun =  torch.nn.MSELoss()

    scores = {k:{name:[] for name in models.keys()} for k in metrics_.keys()}   
    

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)
    for batch, (X, y) in tqdm(enumerate(train_loader)):
        if pp is None:
            X = X.float()
        else:
            X = torch.Tensor(pp.transform(X.detach().numpy())).float()
        y = y.float()
        test_preds, test_losses = ev.test_batch(X, y, models, opts, loss_eval,loss_fun,time=False)
        for i, yi in enumerate(y):
            yi  = yi.detach().numpy()
            preds['y'].append(yi)
            for k,v in test_preds.items(): 
                preds[k].append(v[i])
                for metric_k,metric_v in metrics_.items():
                    score = metric_v[k].update(yi, v[i]).get()
                    #print(score)
                    scores[metric_k][k].append(score)
        
        
    return preds, scores,models,metrics_


def score_evaluate(dataset,models,metrics_,num_its=10000):
    X,y = dataset[0:num_its]
    preds = {name:[] for name in models.keys()}
    preds['y'] = []
    
    scores = {k:{name:[] for name in models.keys()} for k in metrics_.keys()}   
    
    
    for i in tqdm(range(0,num_its)):
        xi = numpy2dict(X[i])
        yi = y[i]
        preds['y'].append(yi)
        
        for name,model in models.items():  
            pred = model.predict_one(xi)
            preds[name].append(pred)
            
            for metric_k,metric_v in metrics_.items():
                score = metric_v[name].update(yi, pred).get()
                if i > 0:
                    scores[metric_k][name].append(score)
        
    
    return preds, scores,metrics_

from collections import deque

class RollingMSE(RegressionMetric):
    def __init__(self,window_size=10000):     
        self.y_true = deque()
        self.y_pred = deque()
        self.sample_weight = deque()
        self.window_size=window_size

    def _eval(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def update(self, y_true, y_pred, sample_weight=1.0):
        if len(self.y_true) == self.window_size:
            self.y_true.popleft()
            self.y_pred.popleft()
            self.sample_weight.popleft()

        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
        self.sample_weight.append(sample_weight)
        
        return self

    def revert(self, y_true, y_pred, sample_weight=1.0):
        return self

    def get(self):
        return mean_squared_error(self.y_true,self.y_pred,sample_weight=self.sample_weight)

class SlidingWindowLR(base.Regressor):
    """
    Wrap our LinearRidge into streaming with a sliding window data structure 
    """

    def __init__(self,window_size=10000,scale=False,ridge=False,ridge_param=1e-2):
        self.X = deque()
        self.y = deque() 
        self.window_size = 10000
        self.sample_weight = None
        
        self.scale=scale
        self.ridge=ridge
        self.ridge_param=ridge_param
        self.lr = None 
        from sk_models import LinearRidge
        
    def learn_one(self,x,y):
        
        if len(self.y) == self.window_size:
            self.y.popleft()
            self.X.popleft()
            #self.sample_weight.popleft()
        
        x = dict2numpyordered(x)
        self.X.append(x)
        self.y.append(y)
        self.lr = LinearRidge(self.scale,self.ridge,self.ridge_param)
        self.lr.fit(self.X,self.y)
    
    def predict_one(self,x):
        if self.lr is None:
            return 0 
        x_arr = [dict2numpyordered(x)]
        pred = self.lr.predict(x_arr)[0]
        return pred
    
class RollingR2(RegressionMetric):
    def __init__(self,window_size=10000):
        self.y_true = deque()
        self.y_pred = deque()
        self.sample_weight = deque()
        self.window_size=window_size

    @property
    def bigger_is_better(self):
        return True

    def update(self, y_true, y_pred, sample_weight=1.0):
        if len(self.y_true) == self.window_size:
            self.y_true.popleft()
            self.y_pred.popleft()
            self.sample_weight.popleft()

        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
        self.sample_weight.append(sample_weight)
        return self

    def revert(self, y_true, y_pred, sample_weight):
        return self

    def get(self):
        if len(self.y_true)  > 1:
            try:
                return r2_score(self.y_true,self.y_pred,sample_weight=self.sample_weight)
            except ZeroDivisionError:
                return 0.0

        # Not defined for n_samples < 2
        return 0.0
    
from river.base import Ensemble
class VotingRegressor(Ensemble):
    
    def __init__(self,models,aggregation_method = "median",record_diversity = False):
        super().__init__(models)
        self.aggregation_method = aggregation_method
        self._MEAN = "mean"
        self._MEDIAN = "median"
        self.n_models = len(models)
        self.disable_weighted_vote = False
        self.record_diversity = record_diversity
        self.diversity_record = None
        
    def learn_one(self, x, y):
        for model in self:
            model.learn_one(x, y)
        return self

    def predict_one(self, x):
        y_pred = np.zeros(self.n_models)
        weights = np.ones(self.n_models)

        for i, model in enumerate(self.models):
            y_pred[i] = model.predict_one(x)
            
        if self.record_diversity:
            if self.diversity_record is None:
                self.diversity_record = {i:[] for i in range(len(self.models))}
                self.diversity_record['v']=[]
            for i, pred in enumerate(y_pred):
                self.diversity_record[i].append(pred)                                   
                                                             

        if self.aggregation_method == self._MEAN:
            voted = np.average(y_pred, weights=weights)
        else:  # self.aggregation_method == self._MEDIAN:
            voted = np.median(y_pred)
                                                                                              
        if self.record_diversity:       
            self.diversity_record['v'].append(voted)     
        return voted
                                                             
                                                             
    def measure_diversity(self,measure='p'):
        """
        return a matrix (we do as a 2d dict)
        """
        if self.diversity_record is None:
            return None
                                                             
        m =  {k1:{k2:'-' for k2 in self.diversity_record.keys()} for k1 in self.diversity_record.keys()}
                                                             
        for k1,v1 in self.diversity_record.items(): 
              for k2,v2 in self.diversity_record.items():                                        
                    p = scipy.stats.pearsonr(v1,v2)[0]
                    if measure == 'p': 
                        m[k1][k2] = p
                    else:                                          
                        m[k1][k2] = -1/2*log(1-p*p,2)
        return pd.DataFrame(m)
        

class BaggingRegressor2(BaggingRegressor):

    def __init__(self,record_diversity=False,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.record_diversity = record_diversity
        self.diversity_record = None
                                                                                                                
    def predict_one(self, x):
        """Averages the predictions of each regressor."""
        preds =  [regressor.predict_one(x) for regressor in self]                                           
        if self.record_diversity:
            if self.diversity_record is None:
                self.diversity_record = {i:[] for i in range(len(self.models))}
                self.diversity_record['v']=[]
            for i, pred in enumerate(preds):
                self.diversity_record[i].append(pred)                            
        
        voted = statistics.median(preds)                                       
        if self.record_diversity:       
            self.diversity_record['v'].append(voted)                                            
                                                             
        return voted

    def measure_diversity(self,measure='p'):
        """
        return a matrix (we do as a 2d dict)
        """
        if self.diversity_record is None:
            return None
                                                             
        m =  {k1:{k2:'-' for k2 in self.diversity_record.keys()} for k1 in self.diversity_record.keys()}
                                                             
        for k1,v1 in self.diversity_record.items(): 
              for k2,v2 in self.diversity_record.items():                                        
                    p = scipy.stats.pearsonr(v1,v2)[0]
                    if measure == 'p': 
                        m[k1][k2] = p 
                    else:                                          
                        m[k1][k2] = -1/2*log(1-p*p,2)
        return pd.DataFrame(m)