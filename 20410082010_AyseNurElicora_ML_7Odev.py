#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# In[9]:


X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])


# In[10]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(X)


# In[11]:


print(kmeans.labels_)
print(kmeans.cluster_centers_)


# In[12]:


#Yukarıdaki sonuçta (X[0], X[1], X[2]) 0 etiketine, son üç sonuç (X[3], X[4], X[5]) ise 1 etiketine ait olmuştur.Küme merkezleri ise, küme 0 için (1, 2) ve küme 1 için (4, 2) olarak bulunmuştur.


# In[13]:


#Sonuçlar
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')


# In[16]:


#küme merkezleri
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# In[ ]:




