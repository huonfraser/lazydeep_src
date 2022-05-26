import torch
from torch import nn
from torch import optim
import torch.functional as F
from skmultiflow.data import FileStream
from transform_stream import TransformStream
import numpy as np
from scipy.io import arff

class EncodeModule(nn.Module):

    def __init__(self):
        pass


class DecodeModule(nn.Module):

    def __init__(self):
        pass


class AutoEncoder(nn.Module):

    def __init__(self,input_size=64,reduce_factor = 1):
        super(AutoEncoder, self).__init__()
        reduce_size = int(input_size*reduce_factor)
        print(reduce_size)

        #network structure here will be really simple, single layer of half the size
        #todo can create seperate models for encode modules
        #tdo then create this as a sequential model (with an OrderedDict)
        #self.encode_layer_hidden = nn.Linear(in_features= input_size, out_features= reduce_size)
        #self.encode_layer_output = nn.Linear(in_features= reduce_size, out_features= reduce_size)
        #self.decode_layer_hidden = nn.Linear(in_features= reduce_size, out_features=reduce_size)
        #self.decode_layer_output = nn.Linear(in_features=reduce_size, out_features=input_size)

        self.encode = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=reduce_size),
            nn.ReLU()
        )
        self.bottle = nn.Sequential(
            nn.Linear(in_features=reduce_size, out_features=reduce_size, bias=False),
            nn.Identity()
        )
        self.decode = nn.Sequential(
            nn.Linear(in_features= reduce_size, out_features=reduce_size),
            nn.ReLU(),
            nn.Linear(in_features=reduce_size, out_features=input_size,bias=False),
        )

        self.model = nn.Sequential(self.encode, self.bottle, self.decode)

    def forward(self,x):
        return self.model.forward(x)

        #activation = self.encode_layer_hidden(x)
        #activation = torch.relu(activation)
        #code = self.encode_layer_output(activation)
        #code = torch.relu(code)
        #activation = self.decode_layer_hidden(code)
        #activation = torch.relu(activation)
        #activation = self.decode_layer_output(activation)
        ##reconstructed = torch.relu(activation)
        #return activation #reconstructed

    def extract_compress(self):
        new_model = nn.Sequential()
        new_model.add(self.encode_layer)
        #model already has its parmeters
        return new_model

    def extract_decompress(self):
        new_model = nn.Sequential(self.decode_layer)
        #model already has its parameters
        return new_model

def train_encoder(data_name,epochs = 1,bs=32,lr=1e-2,loss_fn=nn.MSELoss,opt_fun=optim.SGD,load_type = 'csv'):
    """
    :param data: a generator containing data
    :return:
    """
    #hyperparameters
    epochs = epochs
    bs = bs
    lr = lr

    #define model
    n_inputs = None
    model = None #AutoEncoder()
    loss = loss_fn()
    opt = None

    cached_data = None
    i = 0
    total_loss = 0

    stream = None

    batch_losses = []
    epoch_losses = []

    for epoch in range(0,epochs):
        epoch_loss = 0

        if load_type == 'csv':
            stream = FileStream(data_name,cat_features=[0])
        elif load_type == 'arff':
            arff = arff.loadarff('yeast-train.arff')
            stream = FileStream(arff)

        while stream.has_more_samples():
            X,y = stream.next_sample()
            #setup when reading first data
            if n_inputs == None:
                n_inputs = len(X[0])
                model = AutoEncoder(n_inputs)
                opt = opt_fun(model.parameters(), lr=lr)


            X = np.reshape(np.asarray(X[0]),(1,n_inputs))
           #print(X)
            if cached_data is None:
                cached_data = X
            else:
                cached_data = np.append(cached_data,X,axis=0)

            if len(cached_data) == bs:
                #RUN A BATCH
                cached_data = torch.tensor(cached_data).float()

                #forwards pass
                opt.zero_grad()
                pred = model(cached_data)

                #COMPUTE TRAINING LOSS
                l = loss(pred, cached_data).tolist()[0]
                print(l)
                batch_losses.append(l)
                total_loss += l
                epoch_loss += l
                #print("batch loss {} is: {}".format(i, l))

                l.backward() #COMPUTE GRADIENTS
                opt.step() #UPDATE PARAMETERS BASED ON CURRENT GRADEINT

                #reset bin
                cached_data = None
                i+=1
        print("Epoch {} average loss is {}".format(epoch,epoch_loss/i))
        epoch_losses.append(epoch_loss)

    print("TODO: Write configuration")
    print("-Global average loss is {}".format(total_loss/i/epochs))


    return model

if __name__ == "__main__":

    # define data of length n
    data_name = "data/classification/elecNormNew.csv"
    train_encoder(data_name,epochs=100)



#for each architecture
    # learning rate
    # epochs
    # batch_size

#record batch and average loss
#visualise these