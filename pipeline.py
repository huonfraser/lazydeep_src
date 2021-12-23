from abc import ABC
from sklearn import cross_decomposition as sk_cd
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
    def state(self,fname):
        """
        Return the relevent state of a model (model -> state)
        """
        self.model
    
    @static
    @abstractmethod
    def from_state(self,state):
        """
        Load a model from the given state dict (state -> model)
        """
        pass
    @static
    def save_state(self,fname,state)
        """
        state -> file
        """
        json = jsonpickle.encode(state)
            with open(fname,'w+') as file:
            file.write(json)
    
    @static 
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
    
class PLSRegression(ABC):
    def __init__(self,*args,**kwargs)
        self.model = sk_cd.PLSRegression(*args,**kwargs)
        
    def fit(self,X,y):
        return self.model.fit(X,y)
    
    def transform(self,X,y):
        return self.transform(X,y)
    
    def predict(self,X):
        return self.model.predict(X)
    
    def score(self,X,y,score_fun = mean_squared_error):
        return self.model.score(X,y)
    
    def state(self,fname):
        """
        Take the x_scores,y_scores_
        """
        state_dict = {'x_rotations_':self.model.x_rotations_,
                      'y_rotations_':self.model.y_rotations_,
                      '_x_mean':self.model._x_mean,
                      '_x_std':self.model._x_std,
                      '_y_mean':self.model._y_mean,
                      '_y_std':self.model._y_std:,
                      'coef_':self.model.coef_      
                        }
        return state_dict
    
    def from_state(self,state):
        """
        Load a model from the given state dict (state -> model)
        """
        globals().update(state)

    def reset():
        pass    
        
class CustomWrapper():
    """
    Wrapper to take any sklearn model and wrap it into our framework 
    """

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

    def state(self,fname):
        """
        Return empty dict as want to return nothing
        """
        
        return {}
    
    def from_state(self,state):
        """
        Do nothing 
        """
        pass
            
        
    def reset():
        pass

    

class Pipeline(Learner):
    
    def __init__(*args,**kwargs):
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
        preds = self.predict(X):
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