import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

WineQualityDf = pd.read_csv("../data/WineQualityDf_cleaned.csv")
WineQualityDf = WineQualityDf.drop("Unnamed: 0", axis=1)

X_matrix = WineQualityDf.drop("quality", axis=1).to_numpy()
y = WineQualityDf["quality"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_matrix, y, test_size=0.2, random_state=62)

#Adding intercept
ones_column_train = np.ones(X_train.shape[0])
ones_column_train = ones_column_train.reshape((X_train.shape[0],1))
X_train = np.hstack((ones_column_train, X_train))

ones_column_test = np.ones(X_test.shape[0])
ones_column_test = ones_column_test.reshape((X_test.shape[0],1))
X_test = np.hstack((ones_column_test, X_test))

#Coefficient_matrix

alpha = (np.linalg.inv(X_train.T @ X_train)) @ X_train.T @ y_train

pred_values = np.array(list((map(round, X_test @ alpha))))
error = np.array(list(map(bool, pred_values - y_test)))
error = np.array(list(map(int, ~error)))

accuracy = sum(error) / len(error)

print(f"Model Accuracy = {accuracy * 100:.2f}%")

