#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = sns.load_dataset('iris')


# In[3]:


df.head()


# In[6]:


df['species'].unique()


# In[7]:


df_setosa = df.loc[df['species'] == 'setosa']


# In[17]:


df_versicolor = df.loc[df['species'] == 'versicolor']
df_virginica = df.loc[df['species'] == 'virginica']


# In[20]:


plt.plot(df_setosa['sepal_length'], np.zeros_like(df_setosa['sepal_length']),'o')
plt.plot(df_versicolor['sepal_length'], np.zeros_like(df_versicolor['sepal_length']),'o')
plt.plot(df_virginica['sepal_length'], np.zeros_like(df_virginica['sepal_length']),'o')
plt.xlabel('petal length')
plt.show()


# In[23]:


sns.FacetGrid(df, hue = 'species', size = 5).map(plt.scatter, 'petal_length', 'sepal_width').add_legend()


# In[24]:


sns.pairplot(df, hue = 'species')


# In[ ]:




