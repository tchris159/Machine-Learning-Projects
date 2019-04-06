#Polynomial Regression

# Data Preprocessing Template



#This Project involves a new potential employee who claims 20 years of experience, and made 160k at his previous salary. 
#The HR wants to verify but is only able to find a small data set which appears to have positions with salary increasing exponentially
#As a data scientist I am to verify whether it is truth or bluff through machine learning modeling





# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #want our variable as matrix. Add :2 to the end
y = dataset.iloc[:, 2].values #we want y as a vector

# Splitting the dataset into the Training set and Test set
"""
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
random_state = 0)"""


#splitting would not make sense because it is so small. We will use the entire
# dataset to maximize the accuracy

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Feature scaling is not needed because in the folliwng steps, we use a library 
#which does it for us to yield more accurate results


# Fitting Linear Regression to the dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) 
X_poly = poly_reg.fit_transform(X) 
#poly_reg is a tool that will transform the matrix of features of x into 
#the matrix of features x poly by adding the additional polynomial terms into x
#degree of two means we are only adding one polynomoial term
#fit x first then transform



#Second linear regression object will include the fit that we made with the 
#poly reg object and xpoly features into the linear regression model

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#summary we do a normal linear regression of just X and y
#then we fit this new regression with it associated polynomial features
#finally we fit the new fitted and transformed poly regresion with linear


# Visualizing the Linear Regression results
#The true results for the ten different
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)') 
plt.xlabel('Position level')
plt.ylable('Salary')
plt.show()
#plot X as x axis and predicted for y axis 
#so lin_reg.predict  predicting the X so we use it for the variable

#After modeling the linear Regression we see that it is overall not a good model
#According to the model, we would have offered him 300k




#Visualizing the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1) #create a matrix for more accurate plot
X_grid = X_grid.reshape((len(X_grid), 1)) #we need a vector
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
#we use poly_reg.fit_transform(X)  as the parameter because the original
#was fitted to our X. By doing so we can apply it to all future X's/models
plt.title('Truth or Bluff (Polynomial Regression)') 
plt.xlabel('Position level')
plt.ylable('Salary')
plt.show()

#When we added a third degree it is no longer convex
#The model is even more improved

# Predicting a new result with Linear Regression

lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression

lin_reg_2.predict(poly_reg.fit_transform(6.5)


















