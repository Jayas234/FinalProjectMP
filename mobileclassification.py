import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'D:\Mobile Price Prediction\mobiledata.csv')
print(data.head())

import seaborn as sns
sns.set()
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", linecolor='white', linewidths=1)
plt.show()

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()
lreg.fit(x_train, y_train)

y_pred = lreg.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy of the Logistic Regression Model: ",accuracy)

print(lreg.predict([[1815,0,2.8,0,2,0,33,0.6,159,4,17,607,748,1482,18,0,2,1,0,0]]))
'''
if lreg prediction is 
                0 (low cost)
                1 (medium cost)
                2 (high cost)
                4 (very high cost)
'''






