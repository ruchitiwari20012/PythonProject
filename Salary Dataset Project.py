#!/usr/bin/env python
# coding: utf-8

# In[126]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[127]:


df=pd.read_csv('empl.csv')
print(df)


# In[128]:


df=pd.read_csv('empl.csv',index_col = ['SNo'])
print(df)


# In[129]:


df.dtypes


# In[130]:


import numpy as np
df=pd.read_csv('empl.csv',dtype={'Salary':np.float64})
print(df.dtypes)
df


# In[131]:


import sys 
sys.setrecursionlimit(10000) 


# In[132]:


import numpy as np
ds=df.replace(np.NaN,0)
ds


# In[133]:


ds.head()


# In[134]:


ds.isnull


# In[135]:


ds.isnull().sum()


# In[136]:


ds['Age'].replace(np.NaN,ds['Age'].mean,inplace=True)


# In[137]:


ds


# In[138]:


ds.tail()


# In[139]:


ds['Salary'].max()


# In[140]:


ds['Salary'].min()


# In[141]:


ds['Salary'].plot()


# In[142]:


ds.dtypes


# In[144]:


ds.shape


# In[145]:


ds.describe()


# In[146]:


ds.isnull().sum()


# In[149]:


ds['Age'].replace(np.NaN,ds['Age'].mean,inplace=True)


# In[148]:


ds.isnull().sum()


# In[150]:


ds['City'].replace(np.NaN,ds['City'].mean,inplace=True)


# In[151]:


ds.isnull().sum()


# In[154]:


ds['Salary'].replace(np.NaN,ds['Salary'].mean,inplace=True)


# In[157]:


Salary=ds['Salary']


# In[156]:


ds.isnull().sum()


# In[158]:


sns.heatmap(ds.isnull())


# In[159]:


ds.columns


# In[160]:


corr_hmap=ds.corr()
plt.figure(figsize=(8,7))
sns.heatmap(corr_hmap,annot=True)
plt.show()


# In[161]:


plt.scatter(ds['Age'],ds['Salary'])
plt.show()


# In[162]:


ds.head()


# In[163]:


ds.dtypes


# In[165]:


ds.drop('Name',axis=1,inplace=True)


# In[166]:


ds.head()


# In[167]:


ds.drop('City',axis=1,inplace=True)


# In[168]:


ds


# In[169]:


ds.drop('Country',axis=1,inplace=True)


# In[171]:


ds.head()


# In[172]:


from scipy.stats import zscore
z=np.abs(zscore(ds))
z


# In[174]:


ds_new=ds[(z<3).all(axis=1)]


# In[175]:


ds.shape


# In[176]:


ds_new.shape


# In[177]:


y=ds.iloc[:,-1]


# In[178]:


y


# In[179]:


x=ds.iloc[:,0:2]


# In[180]:


x


# In[182]:


x.shape


# In[183]:


y.shape


# In[184]:


from sklearn.model_selection import train_test_split


# In[185]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=42)


# In[186]:


x_train.shape


# In[187]:


y_train.shape


# In[188]:


x_test.shape


# In[189]:


y_test.shape


# In[190]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[202]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
max_r_score=0
for r_state in range(42,90):
    train_x,test_x,train_y,test_y=train_test_split(x,y,random_state=r_state,test_size=0.20)
    lm=LinearRegression()
    lm.fit(x_train,y_train)
    y_pred=lm.predict(x_test)[:,1]
    r2_scr=r2_score(test_y,y_pred)
    if r2_scr>max_r_score:
        max_r_score=r2_scr
        final_r_state=r_state
        
print()
print()
print("max r2 score corresponding to ",final_r_state,"is",max_r_score)
    


# # Please Rectify This error

# In[203]:


lm=LinearRegression()


# In[204]:


lm.fit(x_train,y_train)


# In[205]:


pred=lm.predict(x_test)
print('PREDICTED salary:',pred)
print('actual salary',y_test)


# In[194]:


print('error:')
print('Mean Absolute error:',mean_absolute_error(y_test,pred))
print('Mean Squared error:',mean_squared_error(y_test,pred))
print('Root mean squared error:',np.sqrt(mean_squared_error(y_test,pred)))


# In[195]:


from sklearn.metrics import r2_score
print(r2_score(y_test,pred))


# In[196]:


from sklearn.linear_model import ElasticNet 
enr=ElasticNet(alpha=0.01)
enr=ElasticNet()
enr.fit(x_train,y_train)
enrpred=enr.predict(x_test)
print(enr.score(x_train,y_train))
enr.coef_


# In[197]:


from sklearn.svm import SVR


# In[198]:


kernellist=['linear','poly','rbf']
for i in kernellist:
    sv=SVR(kernel=i)
    sv.fit(x_train,y_train)
    print(sv.score(x_train,y_train))


# # I know i have done lots of mistake in this dataset but i am unable to solve this issue please clear this to me step by step

# In[ ]:





# In[ ]:





# In[ ]:




