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


df=pd.read_csv('Wine Dataset')


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


df.Class.unique()


# In[11]:


df.Class.value_counts()


# In[12]:


df.isnull().sum()


# In[13]:


sns.heatmap(df.isnull())


# In[14]:


dfcor=df.corr()
dfcor


# In[15]:


sns.heatmap(dfcor)


# In[16]:


df.columns


# In[17]:


df['Proline    '].plot.box()


# In[18]:


df['Malic acid'].plot.box()


# In[19]:


df['Ash'].plot.box()


# In[20]:


df['Alcalinity of ash'].plot.box()


# In[21]:


df['Magnesium'].plot.box()


# In[22]:


df['Total phenols'].plot.box()


# In[23]:


df['Flavanoids'].plot.box()


# In[24]:


df['Nonflavanoid phenols'].plot.box()


# In[25]:


df['Proanthocyanins'].plot.box()


# In[26]:


df.shape


# In[27]:


collist=df.columns.values
ncol=14
nrows=10


# In[28]:


plt.figure(figsize=(ncol,5*ncol))
for i in range(1,len(collist)):
    plt.subplot(nrows,ncol,i+1)
    sns.boxplot(df[collist[i]],color='green',orient='v')
    plt.tight_layout()


# # To check skeweness of the data

# In[29]:


sns.distplot(df['Alcohol'])


# In[30]:


sns.distplot(df['Malic acid'])


# In[31]:


plt.figure(figsize=(16,16))
for i in range (0,len(collist)):
    plt.subplot(nrows,ncol,i+1)
    sns.distplot(df[collist[i]])


# In[32]:


plt.scatter(df['Alcohol'],df['Malic acid'])


# In[33]:


sns.pairplot(df)


# # Removing Outlier

# In[34]:


from scipy.stats import zscore
z=np.abs(zscore(df))
z


# In[35]:


threshold=3
print(np.where(z>3))


# In[36]:


# rows and columns 
z[13][8]


# In[37]:


z[15][5]


# In[38]:


df_new=df[(z<3).all(axis=1)]


# In[39]:


df


# In[40]:


df.shape


# In[41]:


df_new


# 10 rows are deleted as an outlier 

# In[42]:


x=df.iloc[:,1:]


# In[43]:


x


# In[44]:


x.shape


# In[45]:


y=df.iloc[:,0].values


# In[46]:


y


# In[47]:


y.shape


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=.22,random_state=45)


# In[50]:


train_x.shape


# In[51]:


test_x.shape


# In[52]:


train_y.shape


# In[53]:


train_y.shape


# In[54]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x=ss.fit_transform(x)


# In[55]:


from sklearn.linear_model import LogisticRegression


# In[56]:


lg=LogisticRegression()


# In[57]:


lg.fit(train_x,train_y)


# In[58]:


pred=lg.predict(test_x)
print(pred)


# In[59]:


from sklearn.metrics import accuracy_score


# In[60]:


print('accuracy_score:',accuracy_score(test_y,pred))


# In[61]:


from sklearn.metrics import confusion_matrix,classification_report


# In[62]:


print(confusion_matrix(test_y,pred))


# In[63]:


y_pred_prob=lg.predict_proba(test_x)[:,1]


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


# In[67]:


from sklearn.tree import DecisionTreeClassifier


# In[68]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(train_x,train_y)
p=dt.predict(test_x)
print(accuracy_score(test_y,p))


# In[69]:


from sklearn.neighbors import KNeighborsClassifier


# In[70]:


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


# In[71]:


def svmkernel(ker):
    svc=SVC(kernel=ker)
    
    score=cross_val_score(svc,x,y,cv=5)
    print('Mean Scores',score.mean())
    print('standard deviation',score.std())
    y_pred = cross_val_predict(svc,x,y,cv=5)

    conf_mat=confusion_matrix(y,y_pred)
    print(conf_mat)
    


# In[72]:


svmkernel('rbf')


# In[73]:


svmkernel('poly')


# In[74]:


from sklearn.ensemble import RandomForestRegressor


# In[75]:


rf=RandomForestRegressor(n_estimators=200,random_state=45)
rf.fit(train_x,train_y)


# In[76]:


pred=rf.predict(test_x)
pred


# In[77]:


from sklearn.ensemble import AdaBoostRegressor
model= AdaBoostRegressor()
model.fit(train_x,train_y)
print(model.score(train_x,train_y))
abpred=model.predict(test_x)
print(abpred)
model.score(test_x,test_y)


# In[78]:


from sklearn.externals import joblib
joblib.dump(abpred,'abpredsave.obj')


# In[ ]:




