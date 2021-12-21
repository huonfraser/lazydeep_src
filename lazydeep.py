from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from autoencoder import *
from sklearn.base import ClassifierMixin, RegressorMixin
from evaluation import *
from utils import *

class LazyDeepClassifier(ClassifierMixin):

    def __init__(self, extractor = None, n_neighbors=5, pretrained=False, config = None ):
        #extractor paramaters
        self.extractor = extractor
        self.config = config
        self.pretrained = pretrained

        #knn parameters
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def fit(self, X, y,log_file = None):
        """

        :param self:
        :param X:
        :param y:
        :return: The fitted lazydeep classifier
        """

        if self.pretrained: #extractor is fitted
            X = self.extractor.compress(torch.tensor(X).float())
            X = X.detach().numpy()
            self.knn = self.knn.fit(X, y)
        else:
            data = np.append(X,y,axis=1)

            dataset = utils.TabularDataset(data=data, cat_cols=[])
            dl = DataLoader(dataset, bs, shuffle=False, num_workers=1)

            self.extractor = train_extractor(dl,self.extractor,exp,log_file = log_file)
            X = self.extractor.compress(X).detach().numpy() #todo tensor x
            self.knn = self.knn.fit(X, y)

        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
        X = self.extractor.compress(torch.tensor(X).float())
        X = X.detach().numpy()
        return self.knn.predict(X)

    def predict_proba(self,X):
        """

        :param X:
        :return:
        """
        X = torch.tensor(X).float()
        X = self.extractor.compress(X)
        X = X.detach().numpy()
        #todo convert back to array
        return self.knn.predict_proba(X)

    def reset(self):
        self.extractor.reset()


class LazyDeepRegressor(RegressorMixin):

    def __init__(self, extractor = None, regressor = None, pretrained=True, config = None, preprocessing = StandardScaler()):
        #extractor paramaters
        self.extractor = extractor
        self.knn = regressor
        self.config = config
        self.pretrained = pretrained

        self.preprocessing = preprocessing

        #knn parameters


    def fit(self,X,y,log_file = None):
        """

        :param self:
        :param X:
        :param y:
        :return: The fitted lazydeep classifier
        """
        self.preprocessing.fit(X)


        if self.pretrained: #extractor is fitted
            with torch.no_grad():
                X = X.to_numpy()
                X = self.preprocessing.transform(X)
                X = torch.tensor(X).float()

                self.extractor.eval()

                X = self.extractor.compress(X)
                X = X.detach().numpy()
                self.knn = self.knn.fit(X, y)
        else:
            data = X.copy()
            data["target"] = y

            #todo take further split
            trained_models, trained_results = train_extractor(data,{'embedded':self.extractor},self.config,log_file = log_file)

            X = X.to_numpy()
            X = torch.tensor(X).float()

            X = self.extractor.compress(X).detach().numpy()
            self.knn = self.knn.fit(X, y)

        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """

        with torch.no_grad():
            X = X.to_numpy()
            X = self.preprocessing.transform(X)
            X = torch.tensor(X).float()

            #self.extractor.eval()
            X = self.extractor.compress(X)
            X = X.detach().numpy()
        return self.knn.predict(X)

    def predict_proba(self,X):
        """

        :param X:
        :return:
        """
        with torch.no_grad():
            X = torch.tensor(X).float()
            #self.extractor.eval()
            X = self.extractor.compress(X)
            X = X.detach().numpy()
            #todo convert back to array
        return self.knn.predict_proba(X)

    def reset(self):
        self.extractor.reset()
        self.knn.reset()

class NeighbourSelectKNN(RegressorMixin):

    def __init__(self, extractor = None, n_neighbors=5, pretrained=True, config = None ):
        #extractor paramaters
        self.extractor = extractor
        self.config = config
        self.pretrained = pretrained

        #knn parameters
        self.n_neighbors = n_neighbors
        self.knn = KNeighborsRegressor(n_neighbors=self.n_neighbors)




    def fit(self,X,y,log_file = None):
        """

        :param self:
        :param X:
        :param y:
        :return: The fitted lazydeep classifier
        """

        if self.pretrained: #extractor is fitted
            with torch.no_grad():
                X = X.to_numpy()
                X = torch.tensor(X).float()

                self.extractor.eval()

                X = self.extractor.compress(X)
                X = X.detach().numpy()
                self.knn = self.knn.fit(X, y)
        else:
            data = X.copy()
            data["target"] = y

            #todo take further split
            trained_models, trained_results = train_extractor(data,{'embedded':self.extractor},self.config,log_file = log_file)

            X = X.to_numpy()
            X = torch.tensor(X).float()

            X = self.extractor.compress(X).detach().numpy()
            self.knn = self.knn.fit(X, y)

        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
        with torch.no_grad():
            X = X.to_numpy()
            X = torch.tensor(X).float()

            #self.extractor.eval()
            X = self.extractor.compress(X)
            X = X.detach().numpy()
        return self.knn.predict(X)

    def predict_proba(self,X):
        """

        :param X:
        :return:
        """
        with torch.no_grad():
            X = torch.tensor(X).float()
            #self.extractor.eval()
            X = self.extractor.compress(X)
            X = X.detach().numpy()
            #todo convert back to array
        return self.knn.predict_proba(X)

    def reset(self):
        self.extractor.reset()
        self.knn.reset()