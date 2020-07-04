#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[4]:


data=pd.read_csv('sonar.csv',names=range(0,61),header=0)


# In[5]:


data


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.shape


# In[9]:


data.columns


# In[10]:


data.describe()


# In[11]:


data.isnull().sum()


# In[12]:


sns.heatmap(data.isnull())


# In[13]:


datacor=data.corr()
datacor


# In[14]:


sns.heatmap(datacor)


# In[15]:


data.columns


# 

# In[16]:


df=pd.DataFrame(data)


# In[17]:


df


# In[18]:


x=data.iloc[:,0:-1]


# In[19]:


x


# In[20]:


x.shape


# In[21]:


from sklearn.decomposition import PCA


# In[22]:


y=df.iloc[:,-1].values


# In[23]:


y


# # Label Encoder

# In[24]:


from sklearn.preprocessing import LabelEncoder


# In[25]:


le=LabelEncoder()
y=le.fit_transform(y)
y


# # StandardScaler

# In[26]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x=ss.fit_transform(x)


# In[27]:


x


# # Spliting The Data

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.22,random_state=45)


# In[30]:


from sklearn.model_selection import train_test_split,cross_val_score


# In[31]:


print(x_train.shape,x_test.shape)


# In[32]:


print(y_train.shape,y_test.shape)


# # Calling the libraries of Algorithms

# In[33]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# In[34]:


KNN=KNeighborsClassifier()
SV=SVC
LR=LogisticRegression()
DT=DecisionTreeClassifier()
GNB=GaussianNB()


# In[35]:


LR.fit(x_train,y_train)


# In[36]:


pred=LR.predict(x_test)
print(pred)


# In[37]:


print('accuracy_score:',accuracy_score(y_test,pred))


# In[38]:


print(confusion_matrix(y_test,pred))


# In[39]:


le=LabelEncoder()
x=le.fit_transform(y)
x


# In[41]:


y=y.reshape(-1,1)


# In[42]:


y


# In[40]:


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


# In[43]:


x=x.reshape(-1,1)


# In[44]:


x


# In[45]:


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


# In[46]:


from sklearn.tree import DecisionTreeClassifier


# In[47]:


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


#  

# In[48]:


from sklearn.neighbors import KNeighborsClassifier


# In[49]:


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


# In[50]:


def svmkernel(ker):
    svc=SVC(kernel=ker)
    
    score=cross_val_score(svc,x,y,cv=5)
    print('Mean Scores',score.mean())
    print('standard deviation',score.std())
    y_pred = cross_val_predict(svc,x,y,cv=5)

    conf_mat=confusion_matrix(y,y_pred)
    print(conf_mat)
    


# In[51]:


svmkernel('rbf')


# In[52]:


svmkernel('poly')


# In[53]:


from sklearn.ensemble import RandomForestRegressor


# In[54]:


rf=RandomForestRegressor(n_estimators=200,random_state=45)
rf.fit(x_train,y_train)


# In[55]:


pred=rf.predict(x_test)
pred


# In[56]:


from sklearn.ensemble import AdaBoostRegressor
model= AdaBoostRegressor()
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
abpred=model.predict(x_test)
print(abpred)
model.score(x_test,y_test)


# In[57]:


from sklearn.externals import joblib
joblib.dump(abpred,'abpredsave.obj')


# 

# # Completed

# In[51]:


models=[]
models.append(('KNeighborsClassifier',KNN))
models.append(('SVC',SV))
models.append(('LogisticRegression',LR))
models.append(('DecisionTreeClassifier',DT))
models.append(('GaussianNB',GNB))


# In[ ]:





# # Error Is comming 

# In[ ]:


for m in models:
    m.fit(x_train,y_train)
    m.score(x_train,y_train)
    predm=m.predict(x_test)
    print('Accuracy Score of ',m,'is:')
    print(accuracy_score(y_test,predm))
    print(confusion_matrix(y_test,predm))
    print(classification_report(y_test,predm))
    print('\n')


# # With For Loop

# In[ ]:


KNN=KNeighborsClassifier()
SV=SVC
LR=LogisticRegression()
DT=DecisionTreeClassifier()
GNB=GaussianNB()


# # Not able to run in a single for loop

# In[ ]:


Model=[]
score=[]
cvs=[]
rocscore=[]
for name,model in models:
    print('  Name  ')
    print('\n')
    Model.append(name)
    model.fit(x_train,y_train)
    print(model)
    pre=model.predict(x_test)
    print('\n')
    AS=accuracy_score(y_test,pre)
    print('Accuracy_score=',AS)
    score.append(AS*100)
    print('\n')
    sc=cross_val_score(model,x,y,cv=10,scoring='accuracy').mean()
    print('Cross_Val_Score=',sc)
    cvs.append(sc*100)
    print('\n')
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,pre)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    print('roc_auc_score = ',roc_auc)
    rocscore.append(roc_auc* 100)
    print('\n')
    print('classification_report\n',classification_report(y_test,pre))
    print('\n')
    cm=confusion_matrix(y_test,pre)
    print(cm)
    print('\n')
    plt.figure(figsize=(10,40))
    plt.subplot()
    plt.title(name)
    print(sns.heatmap(cm,annot=True))
    plt.subplot()
    plt.title(name)
    plt.plot(false_positive_rate,true_positive_rate,label='AUC=%0.2f'% roc_auc)
    plt.plot([0,1],[0,1],'r--')
    plt.legend(loc='lower right')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    print('\n\n')

