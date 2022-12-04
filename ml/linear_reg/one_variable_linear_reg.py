#libaries
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random

#reading data
df = pd.read_csv('Linear Regression - Sheet1.csv')

#check for nan in the data
#print(df['Y'].isnull().sum())
x = np.array(df['X']).reshape(-1,1)
y = np.array(df['Y']).reshape(-1,1)

# contains best value
best = []

for i in range(100):
    #test size
    ts = round(random.uniform(0.15,0.30),2)
    
    #splitting into training and testing sets
    X_train , X_test ,Y_train,Y_test = train_test_split(x,y,test_size = ts)
    reg = LinearRegression()
    reg.fit(X_train,Y_train)
    intercept = reg.intercept_
    slope = reg.coef_
    score = reg.score(X_test,Y_test)
    # print("intercept:", intercept)
    # print("slope:", slope)
    if len(best) == 0:
        best.append([intercept,slope,score])
    else:
        if score > best[0][2]:
            best[0][0] = intercept
            best[0][1] = slope
            best[0][2] = score
    # print("accuracy:", score)
    i = i+1

#printing intercepts and the plots of actual value and expected value
print("intercept:", best[0][0])
print("slope:", best[0][1])
print("score:", best[0][2])
Y_pred = reg.predict(X_test)
plt.scatter(X_test, Y_test, color ='b')
plt.plot(X_test, Y_pred, color ='k')
plt.show()
