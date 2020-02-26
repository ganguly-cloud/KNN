import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing,neighbors
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
df = pd.read_csv('breast-cancer.csv.csv')

#print df.head()

df.replace('?',-99999,inplace=True) # some missing data is replaced with '?'
#print df .head()

df.drop(['id'],1,inplace=True)
#print df.head()

x =  np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)

clf = neighbors.KNeighborsClassifier()

clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
print y_pred
print type(y_pred)  # <type 'numpy.ndarray'>
'''
[2 2 2 2 2 2 2 2 2 4 4 2 2 2 4 4 4 2 2 4 2 2 2 2 4 4 2 2 4 2 2 2 4 2 4 4 2
 2 2 4 2 4 2 4 2 4 4 2 4 4 4 2 2 2 2 4 4 2 4 2 2 2 2 2 2 2 4 2 2 2 2 4 2 2
 2 2 4 4 2 4 2 2 2 2 2 2 4 2 4 2 2 4 4 2 4 4 4 2 2 4 4 4 2 2 2 2 4 4 4 4 4
 2 2 4 2 2 2 2 2 4 2 4 2 2 2 4 2 4 2 4 2 2 4 2 2 2 2 4 4 2]'''

accuracy = clf.score(x_test,y_test)
print accuracy   # 0.9642857142857143

from sklearn.metrics import classification_report

classfi_report = classification_report(y_test,y_pred)
print classfi_report
'''
              precision    recall  f1-score   support

           2       0.99      0.97      0.98        90
           4       0.94      0.98      0.96        50

   micro avg       0.97      0.97      0.97       140
   macro avg       0.97      0.97      0.97       140
weighted avg       0.97      0.97      0.97       140'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print cm
'''
[[88  3]
 [ 1 48]]'''

# To check or predict a perticular values from dataset or by our own features

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
example_measures =example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print prediction   # [2 2]


# to print Confusion matrix

import seaborn as sns
plt.figsize(figsize=(9,9))
sns.heatmap(cm,annot=True,fmt =".3f",linewidths=.5,square=True,cmap='Blues_r')
plt.xlabel('Predicted label')
plt.ylabel('True label')
title = 'Accuracy Score : {0}'.format(accuracy)
plt.title(title,size = 14)
plt.savefig('CM_After_prediction')
plt.show()
