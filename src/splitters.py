from sklearn.utils import check_random_state, indexable
from sklearn.utils.validation import _num_samples, column_or_1d
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection._split import _BaseKFold

import numpy as np

"""
In this module we are definining splitter classes that define 3 splits, train,test, validate

For cross val we use K2 fold (k fold gives us regular splits)
We also want a leave one out for small datasets 

For train test we want lots of options
    - split of proportions
    - split of fixed numbers
    - split of given indices 

* K2 fold - kfold
*


"""

class TrainTestValSplit():

    def __init__(self, shuffle = False, random_state = None,
                 train_prop = 0.6,
                 test_prop = 0.2,
                 val_prop = 0.2,
                 train_num = None,
                 test_num = None,
                 val_num = None,
                 train_ind = None,
                 test_ind = None,
                 val_ind = None):
        """
        We have 3 sets of parameters corresponding to:
             Random test train val split with proportions
             Random test train val split with numbers
             Random test train val split with indices
        We will take the last defined one as being implemented, in future this could be 3 classes

        :param shuffle:
        :param random_state:
        """
        self.shuffle = shuffle
        self.random_state = random_state

        self.train_prop = train_prop #these 3 are safe
        self.test_prop = test_prop
        self.val_prop = val_prop

        self.train_num = train_num #these 3 can break
        self.test_num = test_num
        self.val_num = val_num

        self.train_ind = train_ind #these 3 can break
        self.test_ind = test_ind
        self.val_ind = val_ind

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        num_instance = _num_samples(X)
        if self.shuffle:
            #find shuffle order
            #reoroder X and y based on shuffle
            pass
        #find split scheme
        if self.train_ind is not None or self.test_ind is not None or self.val_ind is not None:
            #indiices are already given
            #todo check none are non
            train_ind = self.train_ind
            val_ind = self.val_ind
            test_ind = self.test_ind
        elif self.train_num is not None or self.test_num is not None or self.val_num is not None:
            #todo check total numbers < total
            train_ind = indices[0:self.train_num]
            val_ind = indices[self.train_num:self.train_num + self.val_num]
            test_ind = indices[self.train_num + self.val_num:]
        else:
            #todo check proprotions are less than 1
            train_ind = indices[0:int(num_instance*self.train_prop)]
            val_ind = indices[int(num_instance*self.train_prop):int(num_instance*(self.train_prop+self.val_prop))]
            test_ind = indices[int(num_instance*(self.train_prop+self.val_prop)):]

        return train_ind, val_ind, test_ind



class K2Fold(_BaseKFold):

  def __init__(self, n_splits = 5, *, shuffle=False, random_state = None ):
    super().__init__(n_splits = n_splits, shuffle=shuffle,
                     random_state=random_state)
    self.shuffle = shuffle

  def split(self,X,y=None,groups = None):
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for train_index_2, test_index in self._iter_test_masks(X, y, groups):
            train_index_1 = indices[np.logical_and(np.logical_not(test_index),np.logical_not(train_index_2))]
            test_index = indices[test_index]
            train_index_2 = indices[train_index_2]
            yield train_index_1,train_index_2, test_index

  def _iter_test_masks(self, X=None, y=None, groups=None):
      """Generates boolean masks corresponding to test sets.
      By default, delegates to _iter_test_indices(X, y, groups)
      """
      for train_2_index,test_index in self._iter_test_indices(X, y, groups):
          test_mask = np.zeros(_num_samples(X), dtype=bool)
          test_mask[test_index] = True

          train_2_mask = np.zeros(_num_samples(X), dtype=bool)
          train_2_mask[train_2_index] = True
          yield train_2_mask, test_mask


  def _iter_test_indices(self, X, y=None, groups=None):
      n_samples = _num_samples(X)
      indices = np.arange(n_samples)
      if self.shuffle:
          check_random_state(self.random_state).shuffle(indices)

      n_splits = self.n_splits
      fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
      fold_sizes[:n_samples % n_splits] += 1
      current = 0
      for fold_size in fold_sizes:
          start, stop1, stop2 = current, current + fold_size, current + fold_size + fold_size
          if stop2> n_samples:
            yield indices[start:stop1], indices[0:fold_size]
          else:
            yield indices[start:stop1], indices[stop1:stop2]
          current = stop1

class K2TrainTest(_BaseKFold):

  def __init__(self, n_splits = 5, split_num=0, shuffle=False, random_state = None ):
    super().__init__(n_splits = n_splits, shuffle=shuffle,
                     random_state=random_state)
    self.shuffle = shuffle
    self.split_num = split_num

  def split(self,X,y=None,groups = None):
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for i,(train_index_2, test_index) in enumerate(self._iter_test_masks(X, y, groups)):
            if self.split_num == i:
                return_1 = indices[np.logical_and(np.logical_not(test_index),np.logical_not(train_index_2))]
                return_2 = indices[test_index]
                return_3 = indices[train_index_2]
        return return_1, return_2, return_3




  def _iter_test_masks(self, X=None, y=None, groups=None):
      """Generates boolean masks corresponding to test sets.
      By default, delegates to _iter_test_indices(X, y, groups)
      """
      for train_2_index,test_index in self._iter_test_indices(X, y, groups):
          test_mask = np.zeros(_num_samples(X), dtype=bool)
          test_mask[test_index] = True

          train_2_mask = np.zeros(_num_samples(X), dtype=bool)
          train_2_mask[train_2_index] = True
          yield train_2_mask, test_mask


  def _iter_test_indices(self, X, y=None, groups=None):
      n_samples = _num_samples(X)
      indices = np.arange(n_samples)
      if self.shuffle:
          check_random_state(self.random_state).shuffle(indices)

      n_splits = self.n_splits
      fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
      fold_sizes[:n_samples % n_splits] += 1
      current = 0
      for fold_size in fold_sizes:
          start, stop1, stop2 = current, current + fold_size, current + fold_size + fold_size
          if stop2> n_samples:
            yield indices[start:stop1], indices[0:fold_size]
          else:
            yield indices[start:stop1], indices[stop1:stop2]
          current = stop1


