 # # Multiple linear Regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# In[9]:


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
dataset.head()

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3]) #change to numbers binary
onehotencoder = OneHotEncoder(categorical_features = [3]) #change into matrix needed
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results

y_pred = regressor.predict(X_test)

#Builidng the optimal model using Backward Elimination
import statsmodels.formula.api as sm
#this library doesn't assume for a constant where b0*x0 where x0 =1
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#X optimal will be a matrix of only high impact independent variables
regressor_OLS = sm. OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#Remove the highest p value measured by p>|t| on the table summary function

X_opt = X[:, [0, 1, 3, 4, 5]]
#X optimal will be a matrix of only high impact independent variables
regressor_OLS = sm. OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
#X optimal will be a matrix of only high impact independent variables
regressor_OLS = sm. OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
#X optimal will be a matrix of only high impact independent variables
regressor_OLS = sm. OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
#X optimal will be a matrix of only high impact independent variables
regressor_OLS = sm. OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
#X optimal will be a matrix of only high impact independent variables
regressor_OLS = sm. OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


















