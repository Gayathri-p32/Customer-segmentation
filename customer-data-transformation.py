#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import csv


# In[1]:



df=pd.read_csv(r"C:\Users\Sravanti\OneDrive\Desktop\CosFeb.csv")


# In[2]:


df1=pd.read_csv(r"C:\Users\Sravanti\OneDrive\Desktop\CosJan.csv")


# In[3]:


df2=pd.read_csv(r"C:\Users\Sravanti\OneDrive\Desktop\jewelry.csv")


# In[4]:


df3=pd.read_csv(r"C:\Users\Sravanti\OneDrive\Desktop\electronics.csv")


# In[5]:


df.columns


# In[6]:


df1.columns


# In[7]:


df2.columns


# In[8]:


df3.columns


# In[9]:


df.head()


# In[10]:


df['user_id'].unique()


# In[11]:


df.drop(['event_type','user_session'],axis=1,inplace=True)


# In[12]:


df.head()


# In[13]:


df1.drop(['event_type','user_session'],axis=1,inplace=True)


# In[14]:


df1.head()


# In[15]:


df2.drop(['color','metal', 'gem','order_id','gender','quantity'],axis=1,inplace=True)


# In[16]:


df2.head()


# In[17]:


df3.drop(['order_id'],axis=1,inplace=True)


# In[18]:


df3.head()


# In[19]:


frames=[df,df1,df2,df3]


# In[20]:


df4=pd.concat(frames)


# In[21]:


df4.head()


# In[22]:


df4.shape


# In[23]:


df.shape


# In[24]:


df1.shape


# In[25]:


df2.shape


# In[26]:


df3.shape


# In[44]:


df4.to_csv(r'C:\Users\Sravanti\OneDrive\Desktop\merged.csv')


# In[3]:


df4=pd.read_csv(r"C:\Users\Sravanti\OneDrive\Desktop\merged.csv")


# In[9]:


df4.head()


# In[5]:


df4.shape


# In[8]:


df4.drop(['Unnamed: 0'],axis=1,inplace=True)


# In[10]:


df5=pd.read_csv(r"C:\Users\Sravanti\OneDrive\Desktop\CosOct.csv")


# In[11]:


df5.drop(['event_type','user_session'],axis=1,inplace=True)


# In[12]:


df5.head()


# In[13]:


frame=[df5,df4]


# In[14]:


df4=pd.concat(frame)


# In[15]:


df5.shape


# In[18]:


df4.shape


# In[8]:


df4.isna().sum()


# In[41]:


df4.drop(['userid'],axis=1,inplace=True)


# In[3]:


df4=pd.read_csv(r"C:\Users\Sravanti\OneDrive\Desktop\merged.csv")


# In[4]:


df4.head()


# In[5]:


meann=df4['price'].mean()
df4['price'].fillna(meann, inplace=True)


# In[11]:


df4.isna().sum()


# In[7]:


df4.drop(['Unnamed: 0'],axis=1,inplace=True)


# In[8]:


import numpy as np
df4['Quantity']=np.random.randint(1,30,df4.shape[0])


# In[9]:


df4.head()


# In[38]:


import random
def filling(user_id):
    if user_id==0:
        num=random.randint(400000000,800000000)
        return num
    else:
        return user_id


# In[39]:


df4['userid']=df4['user_id'].apply(filling)


# In[42]:


df4.head()


# In[45]:


df4.isna().sum()


# In[35]:


df4[df4['user_id']==0]


# In[34]:


df4['user_id']=df4['user_id'].fillna(0)


# In[ ]:




