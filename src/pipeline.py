from abc import ABC, abstractmethod
from sklearn import cross_decomposition as sk_cd
from sklearn.metrics import mean_squared_error
import jsonpickle
#"we need a way to represent data that works for pls and deep 

class Learner(ABC):
    
    @abstractmethod
    def fit(self,X,y):
        pass
    
    @abstractmethod
    def transform(self,X,y):
        pass
    
    @abstractmethod
    def predict(self,X):
        pass
    
    @abstractmethod
    def score(self,X,y,score_fun = mean_squared_error):
        pass
    
    @abstractmethod
    def state(self):
        """
        Return the relevent state of a model (model -> state)
        """
        pass
    
    @abstractmethod
    def from_state(self,state):
        """
        Load a model from the given state dict (state -> model)
        """
        pass
    
    def save_state(self,state,fname):
        """
        state -> file
        """
        json = jsonpickle.encode(state)
        with open(fname,'w+') as file:
            file.write(json)
    
    def load_state(self,fname):
        """
        file -> state 
        """
        with open(fname,'r') as file:
            text = file.read()
            return jsonpickle.decode(text)
    
    @abstractmethod
    def reset():
        pass
    

class StreamLearner(ABC):
    
    @abstractmethod
    def learn_one(self,X,y):
        pass
    
    @abstractmethod
    def predict_one(self,X):
        pass
    
    
    @abstractmethod
    def state(self):
        """
        Return the relevent state of a model (model -> state)
        """
        pass
    
    @abstractmethod
    def from_state(self,state):
        """
        Load a model from the given state dict (state -> model)
        """
        pass
    
    def save_state(self,state,fname):
        """
        state -> file
        """
        json = jsonpickle.encode(state)
        with open(fname,'w+') as file:
            file.write(json)
    
    def load_state(self,fname):
        """
        file -> state 
        """
        with open(fname,'r') as file:
            text = file.read()
            return jsonpickle.decode(text)
    
    @abstractmethod
    def reset():
        pass
    

class Pipeline(Learner):
    
    
    def __init__(self,*args,**kwargs):
        self.pipes = []
        
        for arg in args:
            self.pipes.append(arg)
            
    def fit(self,X,y):
        for pipe in self.pipes:
            pipe.fit(X,y)
            X,y = pipe.transform(X,y)
            
        return self
    
    def transform(self,X,y):
        for pipe in self.pipes:
            X,y = pipe.transform(X,y)
            
        return X,y
    
    def predict(self, X):
        X,y = self.transform(X,None)
        return y
    
    def score(self, X, y,score_fun = mean_squared_error):
        preds = self.predict(X)
        scores = score_fun(preds,y)
        return scores
        
    def reset(self):
        for pipe in pipes:
            pipe.reset()
            
    def state(self,fname):
        states = [pipe.state() for pipe in self.pipes]
        return jsonpickle.encode(states)
    
    def from_state(self,state):
        """
        We have a nested depickled, so we need to repickle (oppositie of state)
        """
        pickle = jsonpickle.decode(state)
        for i,v in enumerate(pickle):
            self.pipes[i].load_state(v)