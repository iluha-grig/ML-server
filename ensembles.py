import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error as mse


class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.ensemble = []
        self.fitted = False
        self.loss_func = []

    def fit(self, X, y, X_val=None, y_val=None, random_state=0):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        self.ensemble.clear()
        self.loss_func.clear()
        np.random.seed(random_state)
        for i in range(self.n_estimators):
            indexes = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size,
                                         **self.trees_parameters)
            tree.fit(X[indexes], y[indexes])
            self.ensemble.append(tree)
            if i == 0:
                res_tmp = self.ensemble[0].predict(X)[np.newaxis, :]
            else:
                res_tmp = np.vstack((res_tmp, self.ensemble[-1].predict(X)))
            self.loss_func.append(mse(y, np.mean(res_tmp, axis=0)))

        self.fitted = True

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        res1 = self.ensemble[0].predict(X)[np.newaxis, :]
        for i in range(self.n_estimators - 1):
            res1 = np.vstack((res1, self.ensemble[i + 1].predict(X)))
        return np.mean(res1, axis=0)


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.ensemble = []
        self.intercept = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size,
                                               **self.trees_parameters)
        self.coef = np.empty(n_estimators, dtype=np.float64)
        self.fitted = False
        self.loss_func = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        self.ensemble.clear()
        self.loss_func.clear()
        self.intercept.fit(X, y)
        predict = self.intercept.predict(X)
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.feature_subsample_size,
                                         **self.trees_parameters)
            tree.fit(X, y - predict)
            predict_new = tree.predict(X)
            loss = lambda c: 0.5 * np.sum((predict + c * predict_new - y) ** 2)
            self.coef[len(self.ensemble)] = self.learning_rate * minimize_scalar(loss).x
            predict += predict_new * self.coef[len(self.ensemble)]
            self.ensemble.append(tree)
            self.loss_func.append(mse(y, predict))

        self.fitted = True

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        res_intercept = self.intercept.predict(X)
        res1 = self.ensemble[0].predict(X)[np.newaxis, :]
        for i in range(self.n_estimators - 1):
            res1 = np.vstack((res1, self.ensemble[i + 1].predict(X)))
        return res_intercept + np.sum(res1 * self.coef[:, np.newaxis], axis=0)
