#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression


# In[2]:


dataset=pd.read_csv('Data-Train.csv')


# In[5]:


X=dataset['x']
Y=dataset['y']
X1=np.array(X)
Y1=np.array(Y)
Y1=Y1.reshape((-1,1))
X1=X1.reshape((-1,1))


# In[6]:


X1=(X1-X1.mean())/X1.std()
Y1=(Y1-Y1.mean())/Y1.std()


# In[7]:


# visualize
plt.style.use('seaborn')
plt.figure()
plt.scatter(X1,Y1)
plt.title('Normalized Data')
plt.show()


# In[9]:


ones=np.ones((1000,1))
new_X=np.hstack((X1,ones))


# In[10]:


def predict(X1,theta):
    return np.dot(X1,theta)

def closedform(X,Y):
    Y=np.mat(Y)
    first=np.dot(X.T,X)
    second=np.dot(X.T,Y)
    inv=np.linalg.inv(first)
    theta=np.dot(inv,second)
    return theta


# In[11]:


theta=closedform(new_X,Y1)
print('theta=',theta)


# In[12]:


y_train=predict(new_X,theta)


# In[13]:


train_e=(Y1-y_train)
train_error=np.linalg.norm(train_e)


# In[14]:


print('train error=',train_error)


# In[15]:


# visualize
plt.style.use('seaborn')
plt.figure()
plt.scatter(X1,Y1)
plt.plot(X1,predict(new_X,theta),color='red',label='Prediction')
plt.title('Normalized Train Data')
plt.legend()
plt.show()


# In[16]:


test_dataset=pd.read_csv('Data-Test.csv')
X_test=test_dataset['x']
Y_test=test_dataset['y']
X_test=np.array(X_test)
Y_test=np.array(Y_test)
Y_test=Y_test.reshape((-1,1))
X_test=X_test.reshape((-1,1))


# In[17]:


X_test=(X_test-X_test.mean())/X_test.std()
Y_test=(Y_test-Y_test.mean())/Y_test.std()


# In[18]:


# visualize
plt.style.use('seaborn')
plt.figure()
plt.scatter(X_test,Y_test)
plt.title('Normalized Test Data')
plt.show()


# In[19]:


ones=np.ones((300,1))
Xnew_test=np.hstack((X_test,ones))


# In[20]:


y_=predict(Xnew_test,theta)


# In[21]:


test_e=(Y_test-y_)
test_error=np.linalg.norm(test_e)


# In[23]:


print('test error=',test_error)


# In[24]:


# visualize
plt.style.use('seaborn')
plt.figure()
plt.scatter(X_test,Y_test)
plt.plot(X_test,y_,color='red',label='Prediction')
plt.title('Normalized Test Data')
plt.legend()
plt.show()


# In[ ]:




