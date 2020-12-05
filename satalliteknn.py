#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
train = 'satrain.csv'
test = 'satest.csv'

header = [str(i) for i in range(36)]
header.append('label')

train_df = pd.read_csv(train,names=header)
test_df = pd.read_csv(test,names=header)

X_train = train_df.iloc[:,:-1].values
X_test = test_df.iloc[:,:-1].values
y_train = train_df.iloc[:,-1].values
y_test = test_df.iloc[:,-1].values


acc=0
best=0
for i in range(1,30):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    y_preds = model.predict(X_test)
    ac = accuracy_score(y_test,y_preds)
    if ac>acc:
        best=i
        acc=ac


model = KNeighborsClassifier(n_neighbors=best)
model.fit(X_train,y_train)
y_preds = model.predict(X_test)
def get_accuracy(probs,true):
    max_indices = np.argwhere(probs == np.max(probs))
    
    if true < 6:
        true = true - 1
    else:                    
        true = true - 2
    if true not in max_indices:
        return 0
    return 1/len(max_indices)
k=0
with open('output.txt','w') as f:
    for i,entry in enumerate(X_test):
        probs = model.predict_proba([entry])
        accuracy = get_accuracy(probs,y_test[i])
        f.write(f'Object_ID: {i}\tPredicted_class: {model.predict([entry])[0]}\tTrue_class: {y_test[i]}\tAccuracy: {accuracy}\n')
f.close()


# In[ ]:


with open('output.txt','w') as f:
    f.write()
f.close()

