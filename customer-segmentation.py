#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df=pd.read_csv(r"C:/Users/gayat/Desktop/merged.csv")


# In[3]:


df.head()


# In[4]:


df["TotalSum"] = df["Quantity"] * df["price"]


# In[5]:


from datetime import datetime
df['event_time']=pd.to_datetime(df['event_time'],utc=True)


# In[6]:


df.head()


# In[7]:


df.dtypes


# In[8]:


import datetime
snapshot_date = max(df.event_time) + datetime.timedelta(days=1)


# In[9]:


customers = df.groupby(['user_id']).agg({
    'event_time': lambda x: (snapshot_date - x.max()).days,
    'user_id': 'count',
    'TotalSum': 'sum'})


# In[10]:


customers.head()


# In[11]:


customers.rename(columns = {'event_time': 'Recency',
                            'user_id': 'Frequency',
                            'TotalSum': 'MonetaryValue'}, inplace=True)


# In[12]:


customers.head()


# In[33]:


import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import boxcox


data=pd.DataFrame(customers['Recency'])
plt.title("Normal")
res = sn.histplot(data)
plt.show()


# In[38]:


plt.title("Log")
res = sn.histplot(np.log(data))
plt.show()


# In[36]:



plt.title("Sqrt")
res = sn.histplot(np.sqrt(data))
plt.show()


# In[37]:


plt.title("Boxcox")
res = sn.histplot(boxcox(data.iloc[:, 0])[0])
plt.show()


# In[ ]:





# In[ ]:


fig, (ax1, ax2) = plt.subplots(1,2)
data=pd.DataFrame(customers['Frequency'])
plt.title("Normal")
res = sn.histplot(data,ax=ax1)
plt.show()
plt.title("Log")
res = sn.histplot(np.log(data),ax=ax2)
plt.show()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
plt.title("Sqrt")
res = sn.histplot(np.sqrt(data),ax=ax1)
plt.show()
plt.title("Boxcox")
res = sn.histplot(boxcox(data.iloc[:, 0])[0],ax=ax2)
plt.show()


# In[16]:


from scipy.stats import skew
data=pd.DataFrame(customers['Recency'])
l=np.log(data)
s=np.sqrt(data)
bc=boxcox(data.iloc[:, 0])[0]
data1=pd.DataFrame(customers['Frequency'])
lf=np.log(data1)
sf=np.sqrt(data1)
bcf=boxcox(data1.iloc[:, 0])[0]


# In[17]:


print("x\t\tnormal\t\tlog\t\tsqrt\t\tboxcox\n")
print("recency\t{}\t{}\t{}\t{}\n".format(skew(data,axis=0),skew(l,axis=0),skew(s,axis=0),skew(bc,axis=0)))
print("freqncy\t{}\t{}\t{}\t{}\n".format(skew(data1,axis=0),skew(lf,axis=0),skew(sf,axis=0),skew(bcf,axis=0)))


# In[18]:


from scipy import stats
customers_fix = pd.DataFrame()
customers_fix["Recency"] = np.log(customers['Recency'])
customers_fix["Frequency"] = np.log(customers['Frequency'])
customers_fix["MonetaryValue"] = pd.Series(np.cbrt(customers['MonetaryValue'])).values
customers_fix.tail()


# In[19]:


print(customers_fix.mean(axis = 0).round(2)) 
print(customers_fix.std(axis = 0).round(2)) 


# In[20]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(customers_fix)
customers_normalized = scaler.transform(customers_fix)
print(customers_normalized.mean(axis = 0).round(2)) # [0. -0. 0.]
print(customers_normalized.std(axis = 0).round(2))


# In[21]:


customers_fix


# In[22]:


from sklearn.cluster import KMeans
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(customers_normalized)
    sse[k] = kmeans.inertia_ # SSE to closest cluster centroid
plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
sn.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()


# In[23]:


customers_normalized


# In[24]:


model = KMeans(n_clusters=3, random_state=42)
model.fit(customers_normalized)
model.labels_.shape


# In[25]:


customers["Cluster"] = model.labels_
customers.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean', 'count']}).round(2)


# In[26]:


df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
df_normalized['ID'] = customers.index
df_normalized['Cluster'] = model.labels_

df_nor_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['ID', 'Cluster'],
                      value_vars=['Recency','Frequency','MonetaryValue'],
                      var_name='Attribute',
                      value_name='Value')
df_nor_melt.head()

sn.lineplot('Attribute', 'Value', hue='Cluster', data=df_nor_melt,palette='Set1')


# In[27]:


model.predict([[1,0,1]])


# In[ ]:


model.predict([[54,56,24500]])


# In[ ]:





# In[28]:


def customer_name(a):
    n = model.predict(a)
    if(n==0):
        print("New Customer")
    elif(n==1):
        print("Dormant customer")
    else:
        print("Loyal customer")


# In[29]:


customer_name([[54,56,24500]])


# In[30]:


customer_name([[1,0,-200]])


# In[31]:


customer_name([[1,78,100]])


# In[ ]:




