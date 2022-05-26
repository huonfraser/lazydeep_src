from abc import abstractmethod
from copy import deepcopy

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import  mean_squared_error
import logging
import logging.config
import sys
import torch
from collections import defaultdict

from sklearn.linear_model import LinearRegression

def preprocess_subset(subset,pp):
  #extract dataset, make new tabular dataset with
  data = pd.DataFrame([i for i in subset])
  id_cols = subset.dataset.id_cols
  cat_cols = subset.dataset.cat_cols
  output_cols = subset.dataset.output_cols
  ignore_cols = subset.dataset.ignore_cols
  new_dataset = TabularDataset(data,id_cols,cat_cols,output_cols,ignore_cols)
  new_dataset.preprocess(pp)
  return new_dataset

def sample_data(data,random_state):
    nrow, ncol = data.shape
    if nrow > 20000:
        data = data.sample(n=10000,random_state=random_state)
    else:
        data=data.sample(frac=1,random_state=random_state)
        
    return data#.reset_index(drop=True)


class TabularDataset(torch.utils.data.Dataset):

    def __init__(self, data:pd.DataFrame, id_cols = None, cat_cols=None, output_cols=None, ignore_cols= None):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        Parameters
        ----------

        data: pandas data frame
          The data frame object for the input data. It must
          contain all the continuous, categorical and the
          output columns to be used.

        """

        #todo make data be a flexible input, read type and then load to pd if not already

        #todo parse st we leave our data s x,y, turn id_cols,ignOore_cols into seperate databases

        self.nrow, self.ncol = data.shape
        self.num_features = self.ncol-1

        self.id_cols = id_cols
        self.ignore_cols = ignore_cols
        self.cat_cols = cat_cols

        if output_cols is None:
            self.output_cols = [data.columns[self.ncol-1]]
        else:
            self.output_cols = output_cols

        not_x_cols = self.output_cols
        if id_cols is not None:
            not_x_cols = not_x_cols + id_cols
        if ignore_cols is not None:
            not_x_cols = not_x_cols + ignore_cols

        self.x_cols = [i for i in range(0, self.ncol) if
                         not data.columns[i] in not_x_cols]
        self.y_cols = [list(data.columns).index(i) for i in self.output_cols]
        self.meta_cols = [i for i in range(0,self.ncol) if data.columns[i] in id_cols]
        #print(f"meta cols {self.meta_cols}")

        self.x_data = data.iloc[:,self.x_cols]
        self.x_original = deepcopy(self.x_data)
        self.y_data = data.iloc[:,self.y_cols]
        self.meta_data = data.iloc[:,self.meta_cols]
        #print(self.data.columns)

    def split_by_col(self,col = "",train_key="train",val_key="val",test_key="test"):#, preprocessing=None):
        train_ind = self.meta_data.loc[self.meta_data[col] == train_key].index
        val_ind = self.meta_data.loc[self.meta_data[col] == val_key].index
        test_ind = self.meta_data.loc[self.meta_data[col] == test_key].index
        return train_ind,val_ind,test_ind # $@self.split(train_ind,val_ind,test_ind, preprocessing=preprocessing)


    def fit_pp(self,train_ind, preprocessing=None):
        """
        Fir a preprocessor to the relevent partition of data
        """
        if not preprocessing is None:
            preprocessing.fit(self.x_original.iloc[[i for i in train_ind],:],np.squeeze(self.y_data.iloc[[i for i in train_ind],:]))

    def split(self, train_ind,val_ind,test_ind, preprocessing=None):
        """
        Give three partitions of data 
        """
        if not preprocessing is None:
            self.x_data = pd.DataFrame(preprocessing.transform(self.x_original))
        train_data = torch.utils.data.Subset(self,train_ind)
        if not val_ind is None:
            val_data = torch.utils.data.Subset(self,val_ind)
        else:
            val_data = None
        test_data = torch.utils.data.Subset(self,test_ind)

        return train_data,val_data,test_data
    
    def take_summary(self,inds):
        dict_ = {
        'mean': self.x_data.iloc[inds,:].mean(axis='index'),
        'var': self.x_data.iloc[inds,:].var(axis='index'),
        'min': self.x_data.iloc[inds,:].min(axis='index'),
        'lq': self.x_data.iloc[inds,:].quantile(q=0.25,axis='index'),
        'median': self.x_data.iloc[inds,:].median(axis='index'),
        'uq': self.x_data.iloc[inds,:].quantile(q=0.75,axis='index'),
        'max': self.x_data.iloc[inds,:].max(axis='index')
        }
        return dict_
        
    def take_average(self,inds,take_std=False):
        means_ = self.x_data.iloc[inds,:].mean(axis='index')
        if take_std is False:
            return means_
        vars_ = self.x_data.iloc[inds,:].var(axis='index')
        return means_,vars_
    
    def take_median(self,inds,take_std=False):
        medians_ = self.x_data.iloc[inds,:].median(axis='index')
        return medians_
  
    def take_min(self,inds,take_std=False):
        medians_ = self.x_data.iloc[inds,:].min(axis='index')
        return medians_
    
    def take_max(self,inds,take_std=False):
        medians_ = self.x_data.iloc[inds,:].max(axis='index')
        return medians_

    def take_uq(self,inds,take_std=False):
        medians_ = self.x_data.iloc[inds,:].quantile(q=0.75,axis='index')
        return medians_
    
    def take_lq(self,inds,take_std=False):
        medians_ = self.x_data.iloc[inds,:].quantile(q=0.25,axis='index')
        return medians_
    

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.nrow
    
    def get_id_data(self,idx):
        metadata= self.meta_data.iloc[idx,:]

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """

        X = self.x_data.iloc[idx,:].to_numpy().astype(float)
        y = self.y_data.iloc[idx,:].to_numpy().astype(float)
        #print(np.squeeze(X).shape,np.squeeze(y).shape)
        return X,np.squeeze(y)

class PredictorCVResults():

  def __init__(self):
    self.num_folds = 0
    self.df = pd.DataFrame({'fold' : [], 'id': [], 'model_name': [], 'original': [], 'predictions':[]})

  def add_fold(self, predictor_results):
    self.num_folds += 1
    nrow,ncol = predictor_results.df.shape
    predictor_results = predictor_results.df.copy()

    predictor_results["fold"] = [self.num_folds]*nrow

    self.df = pd.concat((self.df,predictor_results),axis=0)
    pass

  def score_by_fold(self):
    score = {}
    folds = self.df["fold"].unique()

    for fold in folds:
      subset = self.df[self.df["fold"]==fold]
      mse = mean_squared_error(subset["original"],subset["prediction"])
      score[folds]=mse
    return score

  def score(self,fold=None,metric = mean_squared_error):
    model_names = self.df["model_name"].unique()
    results = {}
    if fold == None:
      for model_name in model_names:
        subset = self.df[self.df["model_name"] == model_name]
        results[model_name] = metric(subset["original"], subset["predictions"])
    else:
      for model_name in model_names:
        subset = self.df[self.df["model_name"] == model_name]
        subset = subset[subset["fold"] == fold]
        results[model_name] = metric(subset["original"], subset["predictions"])
    return results

  def residuals(self,name=None):

    subset = self.df[self.df["model_name"] == name]
    ax = None
    from matplotlib import cm
    folds = subset["fold"].unique()
    colours = cm.get_cmap('Set1', len(folds))
    for fold in folds:
      subset2 = subset[subset["fold"] == fold]
      if ax is None:
        ax = subset2.plot.scatter(x="original", y="predictions", label=str(fold), color=colours(fold / len(folds)), s=2)
        ax.set_title("{} Residual Plot".format(name))
      else:
        subset2.plot.scatter(x="original", y="predictions", label=str(fold), ax=ax, color=colours(fold / len(folds)),
                            s=2)



  def summary_stats(self,name=None):
    from sklearn.linear_model import LinearRegression
    subset = self.df[self.df["model_name"] == name]

    summary = {}
    residuals = subset["predictions"]-subset["original"]

    from scipy.stats import pearsonr
    summary["correlation coefficient"] = pearsonr(subset["original"],subset["predictions"])[0]

    reg = LinearRegression().fit(subset["original"].to_numpy().reshape(len(subset["original"]),1),subset["predictions"].to_numpy())
    summary["r^2"] = reg.score(subset["original"].to_numpy().reshape(len(subset["original"]),1),subset["predictions"].to_numpy())

    from statistics import stdev
    summary["std dev of residuals"] = stdev(residuals)

    from sklearn.metrics import mean_absolute_error

    summary["MAE"] = mean_absolute_error(subset["predictions"],subset["original"])
    from math import sqrt
    summary["RMSE"] = sqrt(mean_squared_error(subset["predictions"],subset["original"]))

    #todo return table for models

    return(summary)

  def save(self, fname):
    self.df.to_csv(fname)


class PredictorResults():
  def __init__(self):

    self.df = pd.DataFrame({'id': [], 'model_name': [],'original': [],'predictions':[]})

  def add_results(self,id,model_name,original,predictions,accuracy):
    #todo rework with accuracy
    new_df = pd.DataFrame({'id': id, 'model_name': [model_name ]*(len(original)), 'original':original,'predictions':predictions})
    self.df = pd.concat((self.df,new_df))

  def extract_results(self):
    return self.df["original"].to_numpy(),self.df["predictions"].to_numpy()

  def score(self,metric = mean_squared_error):
    model_names = self.df["model_name"].unique()
    results = {}
    for model_name in model_names:
      subset = self.df[self.df["model_name"] == model_name]
      results[model_name] = mean_squared_error(subset["original"], subset["predictions"])
    return results

  def save(self, fname):
    self.df.to_csv(fname)

class ExtractorCVResults():

  def __init__(self):
    """
    Nest extractor test and train results class
    :param n_features:
    :param n_folds:
    """
    self.num_folds = 0
    self.df = pd.DataFrame({'fold':[],'id': [], 'model_name': [],'loss':[]})


  def add_fold(self,results):
    self.num_folds += 1
    nrow, ncol = results.df.shape
    predictor_results = results.df.copy()

    predictor_results["fold"] = [self.num_folds] * nrow

    self.df = pd.concat((self.df, predictor_results), axis=0)

  def fold_summary(self):
    n_folds = len(self.folds)
    summary = None

    for fold in range(n_folds):
      test_split = self.folds[fold][1]
      row = test_split.mean_losses()
      #todo get names
      if summary is None:
        summary = row.reshape(1, len(row))
      else:
        summary = np.append(summary, row.reshape(1, len(row)), 0)

    summary = pd.DataFrame(summary)

    return summary

  def mean_score(self):
    summary = self.fold_summary()
    return summary.mean(axis=1).to_dict()


  def score(self,fold=None,metric = mean_squared_error):
    model_names = self.df["model_name"].unique()
    results = {}
    if fold == None:
      for model_name in model_names:
        subset = self.df[self.df["model_name"] == model_name]
        results[model_name] = subset["loss"].mean()
    else:

      for model_name in model_names:
        subset = self.df[self.df["model_name"] == model_name]
        subset = subset[subset["fold"] == fold]
        results[model_name] = subset["loss"].mean()
    return results

  def save(self, fname):
    self.df.to_csv(fname)

class ExtractorTrainResults():
  """
  Record results for each batch
  """
  def __init__(self,batch_size):
    self.batch_size = batch_size

    self.train_df = pd.DataFrame({'id': [], 'model_name': [],'loss': []})
    self.val_df = pd.DataFrame({'id': [], 'model_name': [],'loss': []})
    self.knn_val_df = pd.DataFrame({'id': [], 'model_name': [],'loss': []})

    self.train_df_to_append = {'id':[],'model_name': [], 'loss': []}
    self.val_df_to_append = {'id':[],'model_name': [], 'loss': []}
    self.knn_val_df_to_append = {'id':[],'model_name': [], 'loss': []}

  def append_all(self):
    self.train_df = pd.concat([self.train_df,pd.DataFrame(self.train_df_to_append)])
    self.val_df = pd.concat([self.val_df, pd.DataFrame(self.val_df_to_append)])
    self.knn_val_df = pd.concat([self.knn_val_df, pd.DataFrame(self.knn_val_df_to_append)])
    self.train_df_to_append = {'id':[],'model_name': [], 'loss': []}
    self.val_df_to_append = {'id':[],'model_name': [], 'loss': []}
    self.knn_val_df_to_append = {'id':[],'model_name': [], 'loss': []}

  def add_train_result(self, id, model_name, loss,bs):
      self.train_df_to_append['id'].extend([id]*bs)
      self.train_df_to_append['model_name'].extend([model_name]*bs)
      self.train_df_to_append['loss'].extend([loss]*bs)


  def add_val_result(self, id, model_name, loss,bs):
      self.val_df_to_append['id'].extend([id]*bs)
      self.val_df_to_append['model_name'].extend([model_name]*bs)
      self.val_df_to_append['loss'].extend([loss]*bs)

  def add_knn_val_result(self, id, model_name, loss):
      self.knn_val_df_to_append['id'].extend([id])
      self.knn_val_df_to_append['model_name'].extend([model_name])
      self.knn_val_df_to_append['loss'].extend([loss])

  def mean_train_losses(self, epoch):
    self.append_all()
    sub = self.train_df.loc[self.train_df['id'] == epoch]
    groups = sub.groupby('model_name', as_index=False)["loss"].mean()
    return groups.set_index('model_name')["loss"].to_dict()

  def mean_val_losses(self, epoch):
    self.append_all()
    sub = self.val_df.loc[self.val_df['id'] == epoch]
    groups = sub.groupby('model_name', as_index=False)["loss"].mean()
    return groups.set_index('model_name')["loss"].to_dict()

  def mean_knn_losses(self, epoch):
    self.append_all()
    sub = self.knn_val_df.loc[self.knn_val_df['id'] == epoch]
    groups = sub.groupby('model_name', as_index=False)["loss"].mean()
    return groups.set_index('model_name')["loss"].to_dict()

  def epoch_losses(self,show_train=False,title=None):
    self.append_all()
    model_names = self.val_df["model_name"].unique()

    val_data = self.val_df.groupby(["id", "model_name"], as_index=False).mean()
    train_data = self.train_df.groupby(["id", "model_name"], as_index=False).mean()
    knn_data = self.knn_val_df.groupby(["id", "model_name"], as_index=False).mean()

    fig, ax = plt.subplots()

    for name in model_names:

        if False:
          subset2 = train_data[train_data["model_name"] == name].dropna()
          ax.plot(subset2["id"],subset2["loss"], label = "{} train".format(name))
        subset1 = val_data[val_data["model_name"] == name].dropna()
        ax.plot(subset1["id"], subset1["loss"], label="{} val".format(name))
        if False:
          subset3 = knn_data[knn_data["model_name"] == name].dropna()
          if not subset3.empty:
            ax.plot(subset1["id"], subset3["loss"], label="{} knn val".format(name))

    ax.legend(loc='upper right',bbox_to_anchor=(1, 0.5))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if title is not None:
      ax.set_title(title)

    pass



def log(str, log_file = None):
  if log_file is not None:
    for f in log_file:
        if f == "console":
          print(str)
        elif f is not None:
            with open(f,'a') as file:
              print(str,file=file)





class ExtractorTestResults():
  """
  Visualisations

  -histogram of losses
  -univariate distributions-including error
  -bivariate distributions-including error

  """

  def __init__(self, n_features):
    self.n_features = n_features
    self.test_size = 0

    #track losses
    self.df = pd.DataFrame({'id': [], 'model_name': [], 'target':[], 'pred':[],'loss': [],'bs':[]})

    #track diff
    self.diffs = pd.DataFrame({'id': [], 'model_name': []})
    for i in range(self.n_features):
      self.diffs["feat_".format(i)] = []



  def add_result(self, id, model_name, y,pred,loss):

    bs = len(pred)
    n_prev_batch = self.test_size/bs

    df2 =  pd.DataFrame({'id': [id]*bs, 'model_name': [model_name]*bs,'target':y, 'pred':pred,'loss':[loss]*bs})

    self.df = pd.concat([self.df,df2])
    self.test_size += bs
    #diff_dict = {'id': id, 'model_name': model_name}
    #for i in range(self.n_features):
    #  n = "feat_{}".format(i)
    #  diff_dict[n] = reproduced[i]-original[i]
    #self.diffs = self.diffs.append([diff_dict])

  def mean_losses(self):
      groups = self.df.groupby('model_name',as_index=False)["loss"].mean()
      return groups.set_index('model_name')["loss"].to_dict()


  def show_losses(self,n_bins=10):
    """
    Show distribution of losses - histogram - scatter - kernal distribution
    :return:
    """
    fig, ax = plt.subplots()
    model_names = self.df["model_name"].unique()

    min = self.df["loss"].min()
    max = self.df["loss"].max()

    bins = [min+i*(max-min)/n_bins for i in range(n_bins+1)]

    for name in model_names:
      subset = self.df[self.df["model_name"] == name]
      ax.hist(x=subset["loss"], bins=bins, alpha=0.5, label=name)
    ax.legend(loc='upper right',bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("MSE Loss")
    fig.show()

  def save(self, fname):
    self.df.to_csv(fname)





  def show_single(self,idx):
    """
    show original, reproduced, diff, line
    :return:
    """
    fig, ax = plt.subplots()
    model_names = self.df["model_name"].unique()
    subset = self.diffs[self.diffs["id"] == idx]
    subset = subset.drop(columns="id")
    subset = subset.drop(columns="feat_")
    x = [i for i in range(self.n_features)]
    for name in model_names:
      row = subset[subset["model_name"] == name]
      row = row.drop(columns="model_name").to_numpy().reshape((self.n_features,)).tolist()
      print(row)
      ax.plot(x, row, label=name)
    ax.legend(loc='upper right',bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Reproduced-Original")

    fig.show()

  def show_diffs(self,idx):
    """
    show arbitarily many differances
    :param idx:
    :return:
    """

    fig, ax = plt.subplots()
    model_names = self.df["model_name"].unique()
    subset = self.diffs[self.diffs["id"]==idx]

    fig.show()


def flip_dicts(dict1):
  flipped = defaultdict(dict)
  for key, val in dict1.items():
    for subkey, subval in val.items():
      flipped[subkey][key] = subval
  return dict(flipped)

class LWRLoss(torch.nn.Module):

  def __init__(self,size_average=None, reduce=None, reduction='mean', n_neighbours=100):
    self.loss = torch.nn.MSELoss(size_average=size_average,reduce=reduce,reduction=reduction)
    self.n_neighbours=n_neighbours
    self.db_X = None
    self.db_Y = None

  def update(self, X, y):
    """
    Update our db
    :param X:
    :param y:
    :return:
    """

  def calc_lwr(self,row):
    #find neighbours
    dist = torch.nn.MSELoss(row, self.db, reduction=None)
    topk = torch.topk(dist, largest=False)
    indices = topk.indices
    #extract neighbours
    X = self.db_X[indices]
    y = self.db_y[indices]

    # train regresion
    lin = LinearRegression()
    lin.fit(X, y)

    #make prediction
    preds = lin.predict(row)
    return preds

  def forward(self, input, target):
    #todo package this up in an apply
    #find indices

    preds = torch.empty(size=(len(input), 1))
    for i, row in enumerate(torch.unbind(input)):
      pred = self.calc_lwr(row)
      preds[i] = pred



    return self.loss(preds, target)

def check_dir(log_dir):
    if not log_dir.exists():
        log_dir.mkdir()
        return log_dir
    else:
        log_dir = ""
        return log_dir


def setup_logger(logger_name="", file_name="", level=logging.INFO): #ContentFormat='%(asctime)s %(levelname)s %(message)s',DateTimeFormat='%Y-%m-%d %H:%M:%S'):
  cfg = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
      f'default_for_{logger_name}': {'format': "%(message)s'"}
    },
    'handlers': {
      f'console_for_{logger_name}': {
        'class': 'logging.StreamHandler',
        'level': level,
        'formatter': f'default_for_{logger_name}',
        'stream': 'ext://sys.stdout'
      },
      f'file_for_{logger_name}': {
        'class': 'logging.FileHandler',
        'level': level,
        'formatter': f'default_for_{logger_name}',
        'filename':f'{file_name}'
        #'args': f'({fname}, a+)' #may need args=('python.log', 'a+') instead
      },
    },
    'loggers': {
      logger_name: {
        'level': level,
        'handlers': [f'console_for_{logger_name}', f'file_for_{logger_name}'],
        'propagate': False
      }
    }
  }
  logging.config.dictConfig(cfg)
  return logging.getLogger(logger_name)

    
def take_summary(data):
        data = pd.DataFrame(data)
        
        dict_ = {
        'mean': data.mean(axis='index'),
        'var': data.var(axis='index'),
        'min': data.min(axis='index'),
        'lq': data.quantile(q=0.25,axis='index'),
        'median': data.median(axis='index'),
        'uq': data.quantile(q=0.75,axis='index'),
        'max': data.max(axis='index')
        }
        return dict_
        
def zip_dict(dict1,dict2):
 
    dict12 = {k:dict1[k]+dict2[k] for k in dict1.keys()}   
    return dict12

def zip_nested_dict(dict1,dict2):
    dict12 = {}
    
    for k in dict1.keys():
        dict12[k] = {name:dict1[k][name]+dict2[k][name] for name in dict1[k].keys()}
    return dict12

def take_subset_by_str(dataset,s,reverse=False):
    col_names = dataset.columns.tolist()
    if reverse:
        encoding = [i for i in col_names if not (s in i)]
    else:
        encoding = [i for i in col_names if (s in i)]
    return dataset[encoding]

def take_subset_by_re(dataset,s,reverse=False):
    col_names = dataset.columns.tolist()
    if reverse:
        encoding = [i for i in col_names if not s.match(i)]
    else:
        encoding = [i for i in col_names if s.match(i)]
    return dataset[encoding]