#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
import warnings 
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('titanic_train.csv')


# In[4]:


df


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.dtypes


# In[8]:


df.columns


# In[9]:


df.describe()


# In[10]:


titanic_cleaned = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
titanic_cleaned.head()


# In[11]:


titanic_cleaned.describe()


# In[12]:


titanic_cleaned.corr(method='pearson')


# In[13]:


titanic_cleaned.columns


# In[14]:


titanic_cleaned.Pclass.unique()


# In[15]:


titanic_cleaned.Pclass.value_counts()


# In[16]:


titanic_cleaned.isnull().sum()


# In[17]:


sns.heatmap(titanic_cleaned.isnull())


# In[18]:


titanic_cleaned['Survived']


# In[19]:


titanic_cleaned['Survived'].value_counts()


# In[20]:


titanic_cleaned['Sex']


# In[21]:


titanic_cleaned['Sex'].value_counts()


# In[22]:


titanic_cleaned.isnull().sum()


# In[23]:


titanic_cleaned['Embarked'].replace(np.NaN,df['Embarked'].mean,inplace=True)


# In[24]:


titanic_cleaned.isnull().sum()


# In[25]:


sns.heatmap(titanic_cleaned.isnull())


# In[26]:


titanic_cleaned.columns


# In[27]:


titanic_cleaned['Survived'].plot.box()


# In[28]:


titanic_cleaned.groupby(['Pclass', 'Sex']).describe()


# In[29]:


sns.countplot('Survived',data=titanic_cleaned)
plt.show()


# In[30]:


titanic_cleaned.groupby(['Sex', 'Survived'])['Survived'].count()


# In[31]:


titanic_cleaned[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
sns.countplot('Sex',hue='Survived',data=titanic_cleaned)
plt.show()


# In[32]:


sns.countplot('Pclass', hue='Survived',data=titanic_cleaned)
plt.title('Pclass: Sruvived vs Dead')
plt.show()


# In[33]:


collist=titanic_cleaned.columns.values
ncol=14
nrows=10


# In[34]:


sns.barplot(x='Pclass', y='Survived', data=titanic_cleaned)


# In[35]:


Sex = pd.get_dummies(titanic_cleaned['Sex'],drop_first=True)

#embark = pd.get_dummies(titanic_cleaned['Embarked'],drop_first=True)


# In[36]:


import sys 
sys.setrecursionlimit(5000) 


# In[37]:


titanic_cleaned


# In[39]:


titanic_cleaned.drop(['Sex','Embarked'],axis=1,inplace=True)


# In[40]:


tc=titanic_cleaned.replace(np.NaN,0)
tc


# In[41]:


tc.isnull().sum()


# In[42]:


x=tc.iloc[:,1:]


# In[43]:


x


# In[44]:


x.shape


# In[45]:


y=tc.iloc[:,0].values


# In[46]:


y.shape


# In[47]:


encoded_data = pd.get_dummies(tc)
encoded_data.head(5)


# In[ ]:





# In[ ]:





# In[48]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
max_r_score=0
for r_state in range(42,90):
    train_x,test_x,train_y,test_y=train_test_split(x,y,random_state=r_state,test_size=0.20)
    lg=LogisticRegression()
    lg.fit(train_x,train_y)
    y_pred_prob=lg.predict_proba(test_x)[:,1]
    pred=lg.predict(test_x)
    print(pred)
    r2_scr=r2_score(test_y,y_pred)
    if r2_scr>max_r_score:
        max_r_score=r2_scr
        final_r_state=r_state
        
print()
print()
print("max r2 score corresponding to ",final_r_state,"is",max_r_score)
    


# # Please rectify my error i am unable to use for loop

# In[50]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.22,random_state=45)


# In[51]:


lg=LogisticRegression()
lg.fit(train_x,train_y)


# In[52]:


pred=lg.predict(test_x)
print(pred)


# In[53]:


train_x.shape


# In[54]:



test_x.shape


# In[55]:


train_y.shape


# In[56]:



test_y.shape


# In[57]:


np.where(x.values >= np.finfo(np.float64).max)


# In[58]:


test_x.fillna(test_x.mean())


# In[60]:


from sklearn.metrics import accuracy_score


# In[61]:


print('accuracy_score:',accuracy_score(test_y,pred))


# In[62]:


from sklearn.metrics import confusion_matrix,classification_report


# In[63]:


print(confusion_matrix(test_y,pred))


# In[ ]:





# In[64]:


y_pred_prob


# In[65]:


from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# In[66]:


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


# In[67]:


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


# In[68]:


from sklearn.tree import DecisionTreeClassifier


# In[69]:


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


# In[70]:


from sklearn.neighbors import KNeighborsClassifier


# In[71]:


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


# In[72]:


def svmkernel(ker):
    svc=SVC(kernel=ker)
    
    score=cross_val_score(svc,x,y,cv=5)
    print('Mean Scores',score.mean())
    print('standard deviation',score.std())
    y_pred = cross_val_predict(svc,x,y,cv=5)

    conf_mat=confusion_matrix(y,y_pred)
    print(conf_mat)
    


# In[73]:


svmkernel('rbf')


# In[74]:


svmkernel('poly')


# In[75]:


from sklearn.ensemble import RandomForestRegressor


# In[76]:


rf=RandomForestRegressor(n_estimators=200,random_state=45)
rf.fit(train_x,train_y)


# In[77]:


pred=rf.predict(test_x)
pred


# In[78]:


from sklearn.ensemble import AdaBoostRegressor
model= AdaBoostRegressor()
model.fit(train_x,train_y)
print(model.score(train_x,train_y))
abpred=model.predict(test_x)
print(abpred)
model.score(test_x,test_y)


# In[79]:


from sklearn.externals import joblib
joblib.dump(y_pred,'y_predsave.obj')


# In[ ]:





# In[ ]:




