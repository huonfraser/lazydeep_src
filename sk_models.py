import jsonpickle
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors._base import _get_weights, _check_weights
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
import warnings
from sklearn.linear_model import Ridge

class CustomWrapper():

    def __init__(self,model):
        self.model = model


    def fit(self,X,y):
        return self.model.fit(X,y)

    def predict(self,X):
        return self.model.predict(X)

    def save(self,fname):
        json = jsonpickle.encode(self.model)
        with open(fname,'w+') as file:
            file.write(json)
            
    def state(self):
        return self
    
    def load_state(self,file):
        return load(file)

    def load(self,fname):
        with open(fname,'r') as file:
            text = file.read()
            return jsonpickle.decode(text)

class LinearRidge():
    def __init__(self,scale=False,ridge=False,ridge_param=1e-2):
        self.scale=scale
        self.ridge=ridge
        self.ridge_param=ridge_param
        if ridge:
            self.linear = Ridge(ridge,ridge_param)
        else:
            self.linear = LinearRegression()
        if scale:
            self.scaler = StandardScaler()

    def fit(self,X,y):
        if self.scale:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        self.linear.fit(X,y)

    def predict(self, X):
        if self.scale:
            X = self.scaler.transform(X)
        preds = self.linear.predict(X)
        return preds

class PLSR():
    def __init__(self,scale=True,n_components=None,ridge=1e-2):
        self.n_components = n_components
        self.pls= PLSRegression(n_components,scale=scale)
        self.linear = Ridge(ridge)
        pass

    def fit(self,X,y):
        self.pls.fit(X,y)
        X_c, y_c = self.pls.transform(X, y)
        self.linear.fit(X_c,y)

    def predict(self, X):
        X_c= self.pls.transform(X)
        preds = self.linear.predict(X_c)
        return preds

class PCR():

    def __init__(self,n_component=4,scale=True, whiten=False, ridge_regression_param = 1e-2):
        self.n_component=n_component
        self.rr_param = ridge_regression_param
        self.scale=scale
        if self.scale:
            self.scaler = StandardScaler()
        self.pca = PCA(n_component,whiten=whiten)
        self.lr = Ridge(ridge_regression_param)

    def fit(self,X,y):
        if self.scale:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        self.pca.fit(X)
        X_reduced = self.pca.transform(X)
        self.lr.fit(X_reduced,y)
        return self

    def predict(self,X):
        if self.scale:
            X = self.scaler.transform(X)
        X_reduced = self.pca.transform(X)
        preds = self.lr.predict(X_reduced)
        return preds

class LocalWeightedRegression():

  def __init__(self,n_neighbours=5,ridge_regression_param = 1e-2,fit_intercept=True, normalize=False, copy_X=False,
                 n_jobs=None, positive=False):
      self.fit_intercept = fit_intercept
      self.normalize = normalize
      self.copy_X = copy_X
      self.n_jobs = n_jobs
      self.positive = positive
      self.ridge_regression_param = ridge_regression_param

      self.n_neighbours = 5
      self._X = None
      self._y = None
      self.kneighbours = LWKNeighborsRegressor(n_neighbors=n_neighbours)

  def fit(self,X,y):
      self._X = X
      self._y = np.asarray(y)
      self.kneighbours.fit(X,y)


  def predict(self,X):
      #import statsmodels.api as sm
      nrow, ncol = X.shape
      preds = []
      distances, indices = self.kneighbours.predict(X)

      for i in range(0,nrow):
        #print(row)
        #fit
        #lm = LinearRegression(fit_intercept=self.fit_intercept,
       #                       normalize=self.normalize,
        #                      copy_X=self.copy_X,
        #                     n_jobs=self.n_jobs)


        X_fit = self._X[indices[i]]
        y_fit = self._y[indices[i]]
        with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          model = Ridge(self.ridge_regression_param)
          model.fit(X_fit, y_fit)
          to_pred = X[i]
          pred = model.predict(to_pred.reshape(1,len(to_pred)))[0]
        preds.append(pred)
      return np.asarray(preds)


class LWKNeighborsRegressor(KNeighborsRegressor):

  def __init__(self, n_neighbors=5, *, weights='uniform',
               algorithm='auto', leaf_size=30,
               p=2, metric='minkowski', metric_params=None, n_jobs=None,
               **kwargs):
    super().__init__(
      n_neighbors=n_neighbors,
      algorithm=algorithm,
      leaf_size=leaf_size, metric=metric, p=p,
      metric_params=metric_params, n_jobs=n_jobs, **kwargs)
    self._X = None

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

class PLSLWR():
    """
    Combine a PLS feature extractor with a locally weighted regression
    """

    def __init__(self, n_components=None, n_neighbours=None):
        self.n_components = n_components
        self.n_neighbours = n_neighbours
        self.pls = PLSRegression(n_components)
        self.lwr = LocalWeightedRegression(n_neighbours=n_neighbours)

    def fit(self, X, y):
        self.pls.fit(X,y)
        X_c, y_c = self.pls.transform(X, y)
        self.lwr.fit(X_c,y)

    def predict(self, X):
        X_c, y_c = self.pls.transform(X)
        preds = self.lwr.predict(X_c)
        return preds

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

def setup_pls_models_slim(n_features):
    values_pls = [20, 50, 100, 200, 500, 1000]
    knn_models = {'lr': CustomWrapper(LinearRidge(ridge_param=1e-2))}
    for value in values_pls:
      if value * 2 < n_features:
        knn_models[f'lwr_k={value}'] = CustomWrapper(LocalWeightedRegression(value))

    return knn_models

def setup_pls_models_exh(n_features):
    values_pls = [i*5 for i in range(1,10)]+[i*50 for i in range(1,21)]
    knn_models = {'lr': CustomWrapper(LinearRidge(ridge_param=1e-2))}
    for value in values_pls:
      if value * 2 < n_features:
        knn_models[f'lwr_k={value}'] = CustomWrapper(LocalWeightedRegression(value))

    return knn_models