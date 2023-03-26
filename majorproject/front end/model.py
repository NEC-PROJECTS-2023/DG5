#!/usr/bin/env python
# coding: utf-8

# ## Importing all libraries

# In[9]:


import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sb
from pandas.plotting import scatter_matrix

# Machine Learning Algorithms
from sklearn.ensemble import RandomForestRegressor

# For Missing Values
from sklearn.impute import SimpleImputer

#For Pickle
import pickle


# ## Importing Dataset

# In[10]:


fifa_raw_dataset = pd.read_csv("C:/Users/Hello/Documents/players_20.csv")


# ## Reading required columns

# In[11]:


features = ['international_reputation', 'overall', 'potential','mentality_composure','age','height_cm','weight_kg','shooting','passing','dribbling','value_eur']
fifa_dataset = fifa_raw_dataset[[*features]]


# ## Replacing Nan Values

# In[12]:


fifa_dataset = fifa_dataset.replace(0,np.nan)


# In[13]:


imputer = SimpleImputer(strategy="median")
imputer.fit(fifa_dataset)
tf = imputer.transform(fifa_dataset)
fifa_dataset_tf = pd.DataFrame(tf, columns=fifa_dataset.columns)


# In[14]:


x = fifa_dataset_tf.drop("value_eur", axis=1)
y = fifa_dataset_tf["value_eur"].copy()


# ## Fitting the model

# In[15]:


forest_reg = RandomForestRegressor(max_features=4, n_estimators=10, random_state=42)
forest_reg.fit(x, y)


# ## Saving model to disk

# In[17]:


pickle.dump(forest_reg, open('model.pkl','wb'))

