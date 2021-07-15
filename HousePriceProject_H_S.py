#PROG8420 – Programming for Big Data             
#Project- Predict House Prices
#Project Members:
#Himali Hemantkumar Thaker (8638034)
#Shubhangini Manoharsinh Zala(8679920)
# Date: April 20th, 2020              		


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import mpl_toolkits
#from sklearn.model_selection import ShuffleSplit

#Reference: https://towardsdatascience.com/machine-learning-project-predicting-boston-house-prices-with-regression-b4e47493633d
#Reference: https://towardsdatascience.com/create-a-model-to-predict-house-prices-using-python-d34fe8fad88f


#%matplotlib inline

_housedata = pd.read_csv("F:/Big Data/Prog for Big Data/kc_house_data.csv")

_housedata.head()
_details=_housedata.describe()
print("All the Data: ",_details)

#Decision tree
_cost = _housedata['price']
_mainfeatures = _housedata.drop('price', axis = 1)

# Success
print("housing dataset has {} data points with {} variables each.".format(*_housedata.shape))

# Minimum price of the data
_min_cost = np.amin(_cost)

# Maximum price of the data
_max_cost = np.amax(_cost)

# Mean price of the data
_mean_cost = np.mean(_cost)

# Median price of the data
_median_cost = np.median(_cost)

# Standard deviation of prices of the data
_stddev_cost = np.std(_cost)

# Show the calculated statistics
print("Statistics for housing dataset:\n")
print("Minimum Cost: ${}".format(_min_cost)) 
print("Maximum Cost: ${}".format(_max_cost))
print("Mean Cost: ${}".format(_mean_cost))
print("Median Cost ${}".format(_median_cost))
print("Standard deviation of Cost: ${}".format(_stddev_cost))


_housedata['bedrooms'].value_counts().plot(kind='bar')
plt.title('Total number of Bedroom')
plt.xlabel('Bedrooms')
plt.ylabel('Count of Bedrooms')
plt.show()
#sns.despine

plt.figure(figsize=(10,10))
sns.jointplot(x=_housedata.lat.values, y=_housedata.long.values, size=10)
plt.ylabel('Longitude of House', fontsize=12)
plt.xlabel('Latitude of House', fontsize=12)
plt.show()
#plt1 = plt()
#sns.despine

plt.scatter(_housedata.price,_housedata.sqft_living)
plt.title("Price of House vs Square Feet of House")
plt.show()

plt.scatter(_housedata.price,_housedata.long)
plt.title("Price of House vs Location of the house area")
plt.show()

plt.scatter(_housedata.price,_housedata.lat)
plt.xlabel("Price of House")
plt.ylabel('Latitude of House')
plt.title("Latitude of House vs Price of House")
plt.show()

plt.scatter(_housedata.bedrooms,_housedata.price)
plt.title("Bedroom of House and Price of House")
plt.xlabel("Bedrooms of House")
plt.ylabel("Price of House")
plt.show()
#sns.despine

plt.scatter((_housedata['sqft_living']+_housedata['sqft_basement']),_housedata['price'])
plt.show()

plt.scatter(_housedata.waterfront,_housedata.price)
plt.title("Waterfront of House [0= no waterfront] vs Price of House")
plt.show()

_lintrain = _housedata.drop(['id', 'price'],axis=1)

_lintrain.head()

_housedata.floors.value_counts().plot(kind='bar')

plt.scatter(_housedata.floors,_housedata.price)
plt.show()

plt.scatter(_housedata.condition,_housedata.price)
plt.show()

plt.scatter(_housedata.zipcode,_housedata.price)
plt.title("costly location by zipcode?")
plt.show()

from sklearn.linear_model import LinearRegression
_linreg = LinearRegression()
_linlabel = _housedata['price']
_conv_dates = [1 if values == 2014 else 0 for values in _housedata.date ]
_housedata['date'] = _conv_dates
_lintrain = _housedata.drop(['id', 'price'],axis=1)

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(_lintrain , _linlabel , test_size = 0.10,random_state =2)

import visuals as vs
vs.ModelLearning(_mainfeatures, _cost)

vs.ModelComplexity(x_train, y_train)

_linreg.fit(x_train,y_train)
_modelres=_linreg.score(x_test,y_test)
print("Accuracy by Liner Regression is  :",_modelres*100)

from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')
clf.fit(x_train, y_train)
_modelaccu=clf.score(x_test,y_test)
print("Accuracy by Gradient Boosting Regressor is :",_modelaccu*100)


#t_sc = np.zeros((params['n_estimators']),dtype=np.float64)
#y_pred = reg.predict(x_test)
#for i,y_pred in enumerate(clf.staged_predict(x_test)):
#    t_sc[i]=clf.loss_(y_test,y_pred)

#testsc = np.arange((params['n_estimators']))+1
#plt.figure(figsize=(12, 6))
#plt.subplot(1, 2, 1)
#plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
#plt.plot(testsc,t_sc,'r-',label = 'set dev test')

#from sklearn.preprocessing import scale
#from sklearn.decomposition import PCA
#pca = PCA()
#pca.fit_transform(scale(train1))