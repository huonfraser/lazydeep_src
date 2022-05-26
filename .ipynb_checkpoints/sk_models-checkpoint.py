import jsonpickle
from sklearn.cross_decomposition import PLSRegression as PLSR
from sklearn.decomposition import PCA as PCA_
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors._base import _get_weights, _check_weights
from sklearn.preprocessing import StandardScaler as SS
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.linear_model import Ridge
from pipeline import Learner
import torch
import math

class CustomWrapper(Learner):

    def __init__(self,model):
        self.model = model


    def fit(self,X,y):
        return self.model.fit(X,y)
    
    def transform(self,X,y):
        #todo, if self.model has a transform function call it, else if has a predict function call it, else??? Also case if both ....
        pass

    def predict(self,X):
        return self.model.predict(X)
    
    def score(self,X,y):
        return self.model.score(X,y)

    def state(self):
        return {}
    
    def from_state(self,fname):
        return self
    
    def reset(self):
        pass

    
class PLSRegression(Learner):
    def __init__(self,*args,**kwargs):
        self.model = PLSR(*args,**kwargs)
        
        self.model.x_rotations_ = np.ndarray([])
        self.model.y_rotations_ = np.ndarray([])
       
        

        self.model.x_weights_ = np.ndarray([])
        self.model.y_weights_ = np.ndarray([])
        self.model.x_loadings_ = np.ndarray([])
        self.model.y_loadings_ = np.ndarray([])
        
        self.model.coef_ = np.ndarray([])
        
        self.model.n_features_in_ = np.ndarray([])
        self.model.n_iter_ = np.ndarray([])
        
        self.model._x_mean = np.ndarray([])
        self.model._x_std = np.ndarray([])
        self.model._y_mean = np.ndarray([])
        self.model._y_std = np.ndarray([])
        
        self.model._x_scores = np.ndarray([])
        self.model._y_scores = np.ndarray([])

        
    def fit(self,X,y):
        self.model.fit(X,y)
    
    def transform(self,X,y=None):
        X = self.model.transform(X)
        return X
    
    def predict(self,X):
        return self.model.predict(X)
    
    def score(self,X,y,score_fun = mean_squared_error):
        return self.model.score(X,y)
    
    def state(self):
        """
        Take the x_scores,y_scores_
        """
        state_dict = {           
            'n_features_in_':self.model.n_features_in_,
            '_x_mean_':self.model._x_mean,
            '_y_mean_':self.model._y_mean,
            '_x_std':self.model._x_std,
            '_y_std':self.model._y_std,
            '_x_scores_':self.model._x_scores,
            '_y_scores_':self.model._y_scores,
            'x_weights_':self.model.x_weights_,
            'y_weights_':self.model.y_weights_,
            'x_loadings_':self.model.x_loadings_,
            'y_loadings_':self.model.y_loadings_,
            'n_iter_':self.model.n_iter_,
            'x_rotations_':self.model.x_rotations_,
            'y_rotations_':self.model.y_rotations_,
            'coef_':self.model.coef_

                        }
        return state_dict
    
    def from_state(self,state):
        """
        Load a model from the given state dict (state -> model)
        """
        for k,v in state.items():
            if k == 'x_mean_':
                setattr(self.model,'_x_mean',v)
            elif k == 'y_mean_':    
                setattr(self.model,'_y_mean',v)
            elif k == 'y_std_':    
                setattr(self.model,'_y_std',v)
            elif k == 'x_std_':    
                setattr(self.model,'_x_std',v)
            elif k == 'y_scores_':    
                setattr(self.model,'_y_scores',v)
            elif k == 'x_scores_':    
                setattr(self.model,'_x_scores',v)
            else:
                setattr(self.model,k,v)     
            
        return self

    def reset():
        pass     
    
class PCA(Learner):
    def __init__(self,*args,**kwargs):
        self.model = PCA_(*args,**kwargs)
        
        self.model.components_ = np.ndarray([])
        self.model.explained_variance_ = np.ndarray([])
        self.model.explained_variance_ratio_ = np.ndarray([])
        self.model.singular_values_ = np.ndarray([])
        self.model.mean_ = np.ndarray([])
        self.model.n_components_ = None
        self.model.n_features_ = None
        self.model_n_samples_ = None
        self.model.noise_variance_ = None
        self.model.n_features_in_ = None 
        
    def fit(self,X,y=None):
        return self.model.fit(X,y)
    
    def transform(self,X,y=None):
        return self.model.transform(X)
    
    def predict(self,X):
        return self.model.predict(X)
    
    def score(self,X,y,score_fun = mean_squared_error):
        return self.model.score(X,y)
    
    def state(self):
        """
        """
        #todo
        state_dict = {'components_':self.model.components_.tolist(),
                     'explained_variance_':self.model.explained_variance_.tolist(),
                     'explained_variance_ratio_':self.model.explained_variance_ratio_.tolist(),
                     'singular_values_':self.model.singular_values_.tolist(),
                     'mean_':self.model.mean_.tolist(),
                     'n_components_':self.model.n_components_,
                     'n_features_':self.model.n_features_,
                     'n_samples_':self.model_n_samples_,
                     'n_samples_':self.model.noise_variance_,
                     'n_features_in_':self.model.n_features_in_}
        return state_dict
    
    def from_state(self,state):
        """
        Load a model from the given state dict (state -> model)
        """
        globals().update(state)
        return self

    def reset():
        pass    

class StandardScaler(Learner):

    def __init__(self, class_name="target"):
        self.class_name = class_name
        self.scaler = SS()
        self.scaler.fit([[0]],[0])

    def fit(self, X,y=None):
        self.scaler.fit(X)

    def transform(self, X,y=None):
        data = self.scaler.transform(X)
        return data

    def predict(self,data):
        pass
        
    def reset(self):
        pass
    
    def score(self,X,y):
        pass

    def state(self):
        state_dict = {'scale_':self.scaler.scale_,
                      'mean_':self.scaler.mean_,
                      'var_':self.scaler.var_,
                      'n_features_in_':self.scaler.n_features_in_,
                      'n_samples_seen_':self.scaler.n_samples_seen_,
                        }
        return state_dict
    
    def from_state(self,state):
        for k,v in state.items():
            setattr(self.scaler,k,v)  
        return self
        
    


class LinearRidge(Learner):
    def __init__(self,scale=False,ridge=False,ridge_param=1e-2):
        self.scale=scale
        self.ridge=ridge
        self.ridge_param=ridge_param
        if ridge:
            self.linear = Ridge(ridge_param)
            #self.linear.fit([[0]],[0])
        else:
            self.linear = LinearRegression()
            #self.linear.fit([[0]],[0])
        if scale:
            self.scaler = StandardScaler()

    def fit(self,X,y):
        if self.scale:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        self.linear.fit(X,y)
        
    def transform(X,y):
        return self.predict(X),y

    def predict(self, X):
        if self.scale:
            X = self.scaler.transform(X)
        preds = self.linear.predict(X)
        return preds
    
    def score(self,X,y):
        pass
    
    def state(self):
        return {}
        state_dict = {'scale':self.scale,
                     'ridge':self.ridge,
                     'ridge_param':self.ridge_param}
        
        if self.scale:
            state_dict["scaler_state"] = self.scaler.state()
       
        state_dict['lin_param'] = {
                'coef_':self.linear.coef_.tolist(),
                'intercept_':self.linear.intercept_
            }
        
        return state_dict
    
    def from_state(self,state):
        return self
    
    def reset():
        pass   

    

class PCR(Learner):

    def __init__(self,n_component=4,scale=True, whiten=False, ridge=False, ridge_regression_param = 1e-2):
        self.n_component=n_component
        self.scale=scale
        if self.scale:
            self.scaler = StandardScaler()
        self.pca = PCA(n_component,whiten=whiten)
        self.lr = LinearRidge(scale=False,ridge=ridge,ridge_param=ridge_regression_param)

    def fit(self,X,y):
        if self.scale:
            self.scaler.fit(X,y)
            X = self.scaler.transform(X,y)
        self.pca.fit(X,y)
        X_reduced = self.pca.transform(X,y)
        self.lr.fit(X_reduced,y)
        return self

    def transform(self,X,y):
        pass
        
    def predict(self,X):
        if self.scale:
            X = self.scaler.transform(X)
        X_reduced = self.pca.transform(X)
        preds = self.lr.predict(X_reduced)
        return preds
    
    def score(self,X,y):
        pass
    
    def state(self):
        state_dict = {
            'n_component':self.n_component,
            'scale':self.scale
        }
        
        if self.scale:
            state_dict["scaler_state"] = self.scaler.state()
            
        state_dict["pca_state"] = self.pca.state()
    
        state_dict["lr_state"] = self.lr.state()
            
        return state_dict
    
    def from_state(self,state):
        return self
    
    def reset():
        pass   

    

class LocalWeightedRegression(Learner):

    def __init__(self,n_neighbours=5,ridge=False,ridge_regression_param = 1e-2,fit_intercept=True, normalize=False, copy_X=False,
                 n_jobs=None, positive=False,floor=True,kernal=True):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        self.ridge = ridge
        self.ridge_regression_param = ridge_regression_param
        self.floor = floor

        self.n_neighbours = 5
        self._X = None
        self._y = None
        self.kneighbours = LWKNeighborsRegressor(n_neighbors=n_neighbours)
        self.kernal=kernal
        
        if self.normalize:
            self.scaler = StandardScaler()

    def fit(self,X,y):
        if self.normalize:
            self.scaler.fit(X,y)
            X = self.scaler.transform(X)
        self._X = X
        self._y = np.asarray(y)
        self.kneighbours.fit(X,y)

    def transform(self,X,y=None):
        pass

    def predict(self,X):
      #import statsmodels.api as sm
        nrow, ncol = X.shape
        preds = []
        
        if self.normalize:
            X = self.scaler.transform(X)
        
        distances, indices = self.kneighbours.predict(X)

        for i in range(0,nrow):
            X_fit = self._X[indices[i]]
            y_fit = self._y[indices[i]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = LinearRidge(ridge=self.ridge,ridge_param=self.ridge_regression_param)
                model.fit(X_fit, y_fit)
                to_pred = X[i]
                pred = model.predict(to_pred.reshape(1,len(to_pred)))[0]
                if self.floor:
                    pred = max(0,pred)
                if self.kernal:
                    if pred < min(y_fit):
                        pred = min(y_fit)
                    elif pred> max(y_fit):
                        pred = max(y_fit)
                    
            preds.append(pred)
        return np.asarray(preds)

    def score(self,x,y):
        pass

    def from_state(self,state):
        pass
    
    def state(self):
        #state_dict = {'ridge':self.ridge,
        #             'ridge_regression_param':self.ridge_regression_param,
        #             'kneighbours_state':self.kneighbours.state()}
        return {}
        return state_dict

    def reset(self):
        self.kneighbours = LWKNeighborsRegressor(n_neighbors=n_neighbours) 


class LWKNeighborsRegressor(KNeighborsRegressor,Learner):

    def __init__(self, n_neighbors=5, *, weights='uniform',
               algorithm='auto', leaf_size=30,
               p=2, metric='minkowski', metric_params=None, n_jobs=None,
               **kwargs):
        super().__init__(
          n_neighbors=n_neighbors,
          algorithm=algorithm,
          leaf_size=leaf_size, metric=metric, p=p,
          metric_params=metric_params, n_jobs=n_jobs, **kwargs)
        self._X = np.ndarray([])

    def predict(self, X):
        """Predict the target for the provided data
        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
              or (n_queries, n_indexed) if metric == 'precomputed'
          Test samples.
        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs), dtype=int
          Target values.
        """
        #prexisiting sklearn code to access indices
        X = check_array(X, accept_sparse='csr')
        neigh_dist, neigh_ind = self.kneighbors(X)

        return neigh_dist, neigh_ind
    
    def transform(self,X,y=None):
        pass
    
    def score(X,y):
        pass

    
    def reset(self):
        pass
    
    def state(self):
        state_dict = {
                      'n_neighbors':self.n_neighbors,
                      '_X': self._X.tolist()
    
        }
        return state_dict
    
    def from_state(self,state):
        pass
    



class PLSLWR(Learner):
    """
    Combine a PLS feature extractor with a locally weighted regression
    """

    def __init__(self, n_components=None, n_neighbours=None):
        self.n_components = n_components
        self.n_neighbours = n_neighbours
        self.pls = PLSR(n_components)
        self.lwr = LocalWeightedRegression(n_neighbours=n_neighbours)

    def fit(self, X, y):
        self.pls.fit(X,y)
        X_c, y_c = self.pls.transform(X, y)
        self.lwr.fit(X_c,y)

    def predict(self, X):
        X_c = self.pls.transform(X)
        preds = self.lwr.predict(X_c)
        return preds
    
    def score(self,X,y):
        pass
    
    def transform(self,X,y=None):
        pass
    
    def state(self):
        state_dict = {'n_components':n_components,
                      'n_neighbours':n_neighbours,
                      'pls_state':self.pls.state(),
                      'lwr_state':self.lwr.state()
        }
        return state_dict
    
    def from_state(self,state):
        pass
    
    def reset():
        pass   

def setup_pls_models(n_features):
    values_pls = [20, 50, 100, 200, 500, 1000]
    values_knn = [1, 5, 10, 20, 50, 100]
    knn_models = {'lr': CustomWrapper(LinearRidge(ridge_param=1e-2))}
    for value in values_knn:
      if value * 2 < n_features:
        knn_models[f'knn_k={value}'] = CustomWrapper(KNeighborsRegressor(value))
    for value in values_pls:
      if value * 2 < n_features:
        knn_models[f'lwr_k={value}'] = CustomWrapper(LocalWeightedRegression(value))

    return knn_models

def setup_pls_models_slim(n_features,kernal=False):
    values_pls = [i*100 for i in range(1,11)]
    knn_models = {}
    for value in values_pls:
        if value * 2 < n_features:
            knn_models[f'lwr_k={value}'] = LocalWeightedRegression(value,kernal=kernal)

    return knn_models

def setup_pls_models_exh(n_features,kernal=False):
    values_pls = [i*10 for i in range(1,6)]+[i*100 for i in range(1,11)]
    knn_models = {'lr': LinearRidge(ridge_param=1e-2)}
    for value in values_pls:
        if value * 2 < n_features:
            knn_models[f'lwr_k={value}'] = LocalWeightedRegression(value)

    return knn_models

class KNNBoost():
    def __init__(self,deep_model,n_neighbors=5, weights='uniform',errors='uniform',convolution='additive',reverse=False):
        self.deep_model = deep_model
        self.n_neighbors = n_neighbors
        

        #boosting params
        self.weights=weights
        self.errors =errors
        self.convolution= convolution
        
        #data to be stored when fitting
        self.kneighbours = None #LWKNeighborsRegressor(n_neighbors=n_neighbours)
        self.deep_errors = None
        self.n_train = 0
        self.deep_features = None
        self.y = None
        self.reverse=reverse
        
    def fit(self,X,y):
        #store (transformed X) and y
        self.deep_features = self.deep_model.compress(torch.Tensor(X).float()).detach().numpy()
        self.y=y
        self.kneighbours = LWKNeighborsRegressor(n_neighbors=self.n_neighbors).fit(self.deep_features,y)

        deep_preds = self.deep_model.forward(torch.Tensor(X).float()).detach().numpy()
        self.deep_errors = np.abs(deep_preds-y)
        self.n_train = len(self.deep_errors)
        
        #print(self.n_train)
        #print(len(self.deep_features))
        return self 
    
    def predict_(self,xi,distances,indices):
        X=self.deep_features[indices]
        y=self.y[indices]
        
        if self.weights=='uniform':
            if self.errors =='uniform':
                #we can just build a non weighted knn 
                weights = np.asarray([1 for i in range(len(y))])
            else: #only errors are triangle
                #scale our errors
                e = self.deep_errors[indices]
                if max(e)<=0:
                    e = np.asarray([1 for i in range(len(y))])
                e = e/np.max(e)     
                if self.reverse:
                    weights = e
                else:
                    weights = (1-e)
                    
        else:#weights are triangle
            if self.errors =='uniform':
                #scale our errors
                if max(distances)<=0: #neighbours not defined
                    distances = np.asarray([1 for i in range(len(y))])
                d = distances/np.max(distances) #distances are now scaled to be in the range (0,1)
                d = (1-d) # invert s.t. 1 is close, 0 is far
                weights = d

            else: #both  are triangle
                e = self.deep_errors[indices]
                if np.max(e)<=0:
                    e = np.asarray([1 for i in range(len(y))])
                e = e/np.max(e) #errors are now scaled to be in the range (0,1)
                if self.reverse:
                    e= (1-e)
                
                if np.max(distances)<=0:  
                    distances = np.asarray([1 for i in range(len(y))])
                d=distances/np.max(distances) #distances are now scaled to be in the range (0,1) #apply_kernal 

                if self.convolution =='additive':
                    weights = (2-e-d)
                else:
                    weights = (1-e)*(1-d)
        
        sum_weights = np.sum(weights)
        if sum_weights <= 0:
            weights = np.asarray([1 for i in range(len(y))])
            sum_weights = np.sum(weights)

        votes = [weights[i]*y[i] for i in range(0,len(y))]
        return np.sum(votes)/sum_weights                   
    
    def predict(self,X):
        #start by transforming into our features 
        Xt = self.deep_model.compress(torch.Tensor(X).float()).detach().numpy()
        nrow, ncol = Xt.shape
        preds = []
        
        distances, indices = self.kneighbours.predict(Xt)

        for i in range(0,nrow):
            pred = self.predict_(Xt[i],distances[i],indices[i])
            preds.append(pred)
        return np.asarray(preds)
    
class LWRBoost():
    def __init__(self,deep_model,n_neighbors=5, weights='uniform',errors='uniform',convolution='additive',kernal=False,reverse=False):
        self.deep_model = deep_model
        self.n_neighbors = n_neighbors
        self.kernal = kernal

        #boosting params
        self.weights=weights
        self.errors =errors
        self.convolution= convolution
        
        #data to be stored when fitting
        self.kneighbours = None #LWKNeighborsRegressor(n_neighbors=n_neighbours)
        self.deep_errors = None
        self.n_train = 0
        self.deep_features = None
        self.y = None
        self.reverse = reverse
        
    def fit(self,X,y):
        #store (transformed X) and y
        self.deep_features = self.deep_model.compress(torch.Tensor(X).float()).detach().numpy()
        self.y=y
        self.kneighbours = LWKNeighborsRegressor(n_neighbors=self.n_neighbors).fit(self.deep_features,y)

        deep_preds = self.deep_model.forward(torch.Tensor(X).float()).detach().numpy()
        self.deep_errors = np.abs(deep_preds-y)
        self.n_train = len(self.deep_errors)
        
        #print(self.n_train)
        #print(len(self.deep_features))
        return self 
    
    def predict_(self,xi,distances,indices):
        X=self.deep_features[indices]
        y=self.y[indices]
        
        
        if self.weights=='uniform':
            if self.errors =='uniform':
                #we can just build a non weighted knn 
                weights = np.asarray([1 for i in range(len(y))])
            else: #only errors are triangle
                #scale our errors
                e = self.deep_errors[indices]
                if max(e)<=0:
                    e = np.asarray([1 for i in range(len(y))])
                e = e/np.max(e)     
                if self.reverse:
                    weights = e
                else:
                    weights = (1-e)
                    
        else:#weights are triangle
            if self.errors =='uniform':
                #scale our errors
                if max(distances)<=0: #neighbours not defined
                    distances = np.asarray([1 for i in range(len(y))])
                d = distances/np.max(distances) #distances are now scaled to be in the range (0,1)
                d = (1-d) # invert s.t. 1 is close, 0 is far
                weights = d

            else: #both  are triangle
                e = self.deep_errors[indices]
                if np.max(e)<=0:
                    e = np.asarray([1 for i in range(len(y))])
                e = e/np.max(e) #errors are now scaled to be in the range (0,1)
                if self.reverse:
                    e= (1-e)
                
                if np.max(distances)<=0:  
                    distances = np.asarray([1 for i in range(len(y))])
                d=distances/np.max(distances) #distances are now scaled to be in the range (0,1) #apply_kernal 

                if self.convolution =='additive':
                    weights = (2-e-d)
                else:
                    weights = (1-e)*(1-d)
        
        sum_weights = np.sum(weights)
        if sum_weights <= 0:
            weights = np.asarray([1 for i in range(len(y))])
            sum_weights = np.sum(weights)
        
        model = LinearRegression()
        model.fit(X, y,sample_weight=weights)
        pred = model.predict(xi.reshape(1,-1))[0]
        if self.kernal:
            if pred < min(y_fit):
                pred = min(y_fit)
            elif pred> max(y_fit):
                pred = max(y_fit) 
        
        return pred                    
    
    def predict(self,X):
        #start by transforming into our features 
        Xt = self.deep_model.compress(torch.Tensor(X).float()).detach().numpy()
        nrow, ncol = Xt.shape
        preds = []
        
        distances, indices = self.kneighbours.predict(Xt)

        for i in range(0,nrow):
            pred = self.predict_(Xt[i],distances[i],indices[i])
            preds.append(pred)
        return np.asarray(preds)
    
    
    
class DeepKNN(KNeighborsRegressor):
    
    def __init__(self, *args, **kwargs):
        self.deep_model = kwargs.pop('deep')
        super(DeepKNN,self).__init__(*args,**kwargs)
        
        
    def predict(self, X):
        Xt = self.deep_model.compress(torch.tensor(X).float).detach().numpy()
        return super().predict(Xt)
    
    def fit(self,X,y):
        Xt = self.deep_model.compress(torch.tensor(X).float).detach().numpy()
        return super().fit(Xt,y)
        