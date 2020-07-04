#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[3]:


data=pd.read_csv('agaricus-lepiota.data',names=range(0,23),header=0)


# In[4]:


data


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


data.shape


# In[8]:


data.columns


# In[9]:


encoded_data = pd.get_dummies(data)
encoded_data.head(5)


# In[10]:


data.describe()


# In[11]:


data.isnull().sum()


# In[12]:


sns.heatmap(data.isnull())


# In[13]:


df=pd.DataFrame(data)


# In[14]:


df


# In[15]:


x=data.iloc[:,1:]


# In[16]:


x


# In[17]:


x.shape


# In[18]:


y=data.iloc[:,0].values


# In[19]:


y


# In[20]:


from sklearn.preprocessing import LabelEncoder


# In[21]:


le=LabelEncoder()
y=le.fit_transform(y)
y


# In[22]:


le=LabelEncoder()
x=le.fit_transform(y)
x


# In[23]:


y


# In[24]:


y=y.reshape(-1,1)


# In[25]:


from sklearn.model_selection import train_test_split,cross_val_score


# In[26]:


x=x.reshape(-1,1)


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.22,random_state=45)


# In[28]:


print(x_train.shape,x_test.shape)


# In[29]:


print(y_train.shape,y_test.shape)


# In[30]:


y_train


# In[31]:


y_test


# In[32]:


x_train


# In[33]:


x_test


# In[34]:


from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression 


# In[35]:


lg=LogisticRegression()
lg.fit(x_train,y_train)


# In[36]:


pred=lg.predict(x_test)
print(pred)


# In[37]:


print('accuracy_score:',accuracy_score(y_test,pred))


# In[38]:


y_pred_prob=lg.predict_proba(x_test)[:,1]


# In[39]:


y_pred_prob


# In[40]:


from sklearn.metrics import confusion_matrix

mnb=MultinomialNB()

score=cross_val_score(mnb,x,y,cv=5)
print('score:',score)

print('Mean Scores',score.mean())
print('standard deviation',score.std())
y_pred = cross_val_predict(mnb,x,y,cv=5)

print('y prediction values')
print(y_pred)

conf_mat=confusion_matrix(y,y_pred)
conf_mat


# In[41]:


print(confusion_matrix(y_test,pred))


# In[42]:


sv=SVC()
score=cross_val_score(sv,x,y,cv=5,scoring='accuracy')
print('accuracy score=',score)

print('Mean Scores',score.mean())
print('standard deviation',score.std())
y_pred = cross_val_predict(sv,x,y,cv=5)

print('y prediction values')
print(y_pred)

conf_mat=confusion_matrix(y,y_pred)
conf_mat

from sklearn.metrics import accuracy_score
accuracy_score(y,y_pred)


# In[43]:


dtc=DecisionTreeClassifier(criterion='gini')
score=cross_val_score(dtc,x,y,cv=5,scoring='f1_macro')
print('F1-score:',score)
print('Mean Scores',score.mean())
print('standard deviation',score.std())
y_pred = cross_val_predict(dtc,x,y,cv=5)

print('y prediction values')
print(y_pred)

conf_mat=confusion_matrix(y,y_pred)
conf_mat


# In[44]:


from sklearn.neighbors import KNeighborsClassifier


# In[45]:


knn=KNeighborsClassifier()
score=cross_val_score(knn,x,y,cv=10,scoring='accuracy')
print('Accuracy score:',score)
print('Mean Scores',score.mean())
print('standard deviation',score.std())
y_pred = cross_val_predict(knn,x,y,cv=5)

print('y prediction values')
print(y_pred)

print('y_pred.shape',y_pred.shape)

conf_mat=confusion_matrix(y,y_pred)
conf_mat


# In[46]:


def svmkernel(ker):
    svc=SVC(kernel=ker)
    
    score=cross_val_score(svc,x,y,cv=5)
    print('Mean Scores',score.mean())
    print('standard deviation',score.std())
    y_pred = cross_val_predict(svc,x,y,cv=5)

    conf_mat=confusion_matrix(y,y_pred)
    print(conf_mat)
    


# In[47]:


svmkernel('rbf')


# In[48]:


svmkernel('poly')


# In[49]:


from sklearn.ensemble import RandomForestRegressor


# In[50]:


rf=RandomForestRegressor(n_estimators=200,random_state=45)
rf.fit(x_train,y_train)


# In[51]:


pred=rf.predict(x_test)
pred


# In[52]:


from sklearn.ensemble import AdaBoostRegressor
model= AdaBoostRegressor()
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
abpred=model.predict(x_test)
print(abpred)
model.score(x_test,y_test)


# In[53]:


from sklearn.externals import joblib
joblib.dump(abpred,'abpredsave.obj')


# In[ ]:




