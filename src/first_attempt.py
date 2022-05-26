import torch
from torch import nn
from torch import optim
import torch.functional as F
import river as R
from stream_autoencoder import AutoEncoder

class AutoEncoderWrapper(R.BaseClassifier):


    """
    KNNRegressor on features extracted with an autoencoder

    """
    def __init__(self,input_size = 64, bs=32, lr=1e-6):
        #setup hyperparameters, n_epochs = 1
        self.bs = bs
        self.lr = lr

        #task parameters
        self.input_size = input_size

        self.model = AutoEncoder()
        self.knn = R.KNNRegressor()
        self.loss_func = F.mse_loss
        self.opt = optim.SGD(self.model.parameters(), lr=self.lr)

        self.cached_x = []
        pass

    def train_autoencoder(self,x):
        # train
        self.cached_x.append(x)
        if len(self.cached_x) == self.bs:
            for i in range((n - 1) // bs + 1):
                start_i = i * bs
                end_i = start_i + bs
                x = data[start_i:end_i]

                pred = self.model(x)
                loss = self.loss_func(pred, x)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            self.cached_x = []
            self.cached_y = []

    def train_knn(self,x,y):
        self.knn.train_one(x,y)

    def learn_one(self, x,y):
        #add to cache_data

        self.train_autoencoder(x)
        encoding = self.model.extract_compress().forward(x) #today parse forward
        self.train_knn(encoding,y)

        #todo store test metrics
        pass
        
    def predict_one(self,x,y):
        pass

    def reset(self):
        pass

    def clone(self):
        pass




#define data of length n
data_name = "data/abalone.csv"
n_inputs = None

import csv
with open(data_name) as file:
    reader = csv.reader(file)
    for row in reader:
        if n_inputs == None:
            n_input = (len(row)-1)
            model =
        data = row[0:n_input]

#hyperparameters
epochs = 1
bs = 32
lr = 0.00001


#Learner


