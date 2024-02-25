#!/usr/bin/env python
# coding: utf-8

# # Adult income data analysis by KODI VENU

# In[1]:


import pandas as pd
import numpy as np
data = pd.read_csv('adult.csv')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# Displaying Top 10 Rows of the Dataset

# In[2]:


data.head(10)


# Displaying Last 10 Rows of the Dataset

# In[3]:


data.tail(10)


# Finding shape of the Dataset

# In[4]:


data.shape


# In[5]:


print("Number of rows", data.shape[0])
print("Number of columns", data.shape[1])


# In[6]:


data.info()


# Fetching random sample from the dataset(50%)

# In[7]:


data.sample(frac=0.50,random_state=42)


# In[8]:


data.sample(frac=0.50)


# In[9]:


data1=data.sample(frac=0.50,random_state=42)


# In[10]:


data1


# In[11]:


data.head(1)


# Checking Null Values in the dataset

# In[12]:


data.isnull().sum()


# In[13]:


sns.heatmap(data.isnull())


# Perform data cleaning Replacing '?' with NaN

# In[14]:


data.tail(20)


# In[15]:


data.isin(['?'])


# In[16]:


data.isin(['?']).sum()


# In[17]:


data.columns


# In[18]:


data['workclass']=data['workclass'].replace('?',np.nan)


# In[19]:


data['occupation']=data['occupation'].replace('?',np.nan)
data['native-country']=data['native-country'].replace('?',np.nan)


# In[20]:


data.tail(20)


# In[21]:


data.isin(['?']).sum()


# In[22]:


data.isnull().sum()


# In[23]:


sns.heatmap(data.isnull())


# Drop all the missing values

# In[24]:


per_missing=data.isnull().sum()*100/len(data)


# In[25]:


per_missing


# In[26]:


data.dropna(how='any',inplace=True)
data.shape


# check for duplicate data and drop them

# In[27]:


dup=data.duplicated().any()


# In[28]:


print('Are there any duplicated values in data', dup)


# In[29]:


data=data.drop_duplicates()


# In[30]:


data.shape


# Dataset Overall Statistics

# In[31]:


data.describe()


# In[32]:


data.describe(include='all')


# Drop the columns - education_num, capital_gain, capital_loss

# In[33]:


data.columns


# In[34]:


data['education'].unique()


# In[35]:


data['educational-num'].unique()


# In[36]:


data.columns


# In[37]:


data=data.drop(['educational-num','capital-gain','capital-loss'],axis=1)


# In[38]:


data.columns


# what is the distribution of Age column?

# In[39]:


data.columns


# In[40]:


data['age'].describe()


# In[41]:


data['age'].hist()


# Find total number of persons having age between 17 to 48 (inclusive) using between method

# In[42]:


data.columns


# In[43]:


(data['age']>=17) & (data['age']<=48)


# In[44]:


sum((data['age']>=17) & (data['age']<=48))


# In[45]:


sum(data['age'].between(17,48))


# What is the distribution of workclass column?

# In[46]:


data.columns


# In[47]:


data['workclass'].describe()


# In[48]:


data['workclass'].hist()
plt.figure(figsize=(45,92))
plt.show()


# How many persons having the bachelors or masters degree?

# In[49]:


data.columns


# In[50]:


data['education'].head(30)


# In[51]:


filter1 = data['education']=='Bachelors'
filter2 = data['education']=='Masters'


# In[52]:


data[filter1 | filter2 ]


# In[53]:


len(data[filter1 | filter2 ])


# In[54]:


data['education'].isin(['Bachelors','Masters'])


# In[55]:


sum(data['education'].isin(['Bachelors','Masters']))


# Bivariate analysis

# In[56]:


data.columns


# In[57]:


data.rename(columns={'income':"Salary"},inplace=True)


# In[58]:


data.columns


# In[59]:


sns.boxplot(x='Salary',y='age',data=data)


# Replace salary values['<=50K','>=50K'] with 0 and 1

# In[60]:


data.columns


# In[61]:


data['Salary'].unique()


# In[62]:


data['Salary'].value_counts()


# In[63]:


sns.countplot('Salary',data=data)


# In[65]:


data['Salary']=data['Salary'].map({'<=50K':0,'>50K':1})


# In[66]:


data.head()


# In[68]:


data['Salary'].value_counts()


# Which workclass getting the highest salary? 

# In[69]:


data.columns


# In[71]:


data.groupby('workclass')['Salary'].mean()


# In[72]:


data.groupby('workclass')['Salary'].mean().sort_values(ascending=False)


# How has better chance to get salary greater than 50K Male or Female?

# In[73]:


data.columns


# In[74]:


data.groupby('gender')['Salary'].mean()


# In[75]:


data.groupby('gender')['Salary'].mean().sort_values(ascending=False)


# Convert workclass columns datatype to category datatype

# In[76]:


data.info()


# In[77]:


data['workclass'].astype('category')


# In[78]:


data['workclass']=data['workclass'].astype('category')


# In[79]:


data.info()


# In[ ]:




