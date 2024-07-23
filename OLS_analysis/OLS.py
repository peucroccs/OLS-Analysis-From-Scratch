import numpy as np 
import pandas as pd 
from scipy.stats import t

class OLS():
    def fit(self, X_train, y_train):
        #Initiate dataset
        self.X_train = X_train
        self.y_train = y_train

        #Find the coefficients and R2
        self.coefficients = (np.linalg.inv(X_train.T @ X_train)) @ X_train.T @ y_train
        prediction_values = X_train @ self.coefficients
        residual_array = y_train - prediction_values

        rss = np.sum(residual_array**2)
        tss = np.sum((y_train - np.mean(y_train))**2)
        self.R2 = 1 - (rss / tss)
        self.d_freedom = X_train.shape[0] - X_train.shape[1]

        self.adj_R2 = 1 - ((1-self.R2)*(X_train.shape[0] - 1))/(self.d_freedom- 1)

        #Find the statistics of the regression

        sigma = rss / self.d_freedom
        self.coeff_standard_errors = (sigma**0.5)*(np.diag(np.linalg.inv(X_train.T @ X_train))**0.5)

        self.t_scores = self.coefficients / self.coeff_standard_errors

        p_values = []
        for t_v in self.t_scores:
            p = 2*(1-t.cdf(np.abs(t_v), self.d_freedom))
            p_values.append(p)
        self.p_values = np.array(p_values)

    def predict(self, X):
        return X @ self.coefficients
    
    def summary(self):
        print("OLS Regression Results")
        print(f"Name | Coeff Value | C_ste | t | p")
        for i in range(len(self.coefficients)):
            print(f"Beta{i} | {self.coefficients[i]} | {self.coeff_standard_errors[i]} | {self.t_scores[i]} | {self.p_values[i]}")