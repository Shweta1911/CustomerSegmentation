#!/usr/bin/env python
# coding: utf-8

# # Loading Dependencies

# In[1]:


import pandas as pd


# # Loading Dataset

# In[2]:


df = pd.read_excel('Online Retail.xlsx')
print(df.head())


# # Data Processing

# ### Eliminating data with item returns (negative quantity)

# In[3]:


df = df.loc[df['Quantity'] > 0]


# ### Identify null components

# In[4]:


df.info()


# ### Handling Nan CustomerID

# In[5]:


df['CustomerID'].isna().sum()
df = df.dropna(subset=['CustomerID'])


# ### Creating the customer-item matrix

# In[6]:


customer_item_matrix = df.pivot_table(
    index='CustomerID',
    columns='StockCode',
    values='Quantity',
    aggfunc='sum'
)
customer_item_matrix.loc[12481:].head()


# In[7]:


print(customer_item_matrix.shape)
customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)


# # Collaborative Filtering

# In[8]:


from sklearn.metrics.pairwise import cosine_similarity


# ## User based collaborative filtering

# In[9]:


user_user_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix))
user_user_sim_matrix.head()


# In[10]:


#Renaming index and column names

user_user_sim_matrix.columns = customer_item_matrix.index

user_user_sim_matrix['CustomerID'] = customer_item_matrix.index
user_user_sim_matrix = user_user_sim_matrix.set_index('CustomerID')
user_user_sim_matrix.head()


# In[11]:


user_user_sim_matrix.loc[12350.0].sort_values(ascending=False).head(10)


# ### Making Recommendations

# In[12]:


user_user_sim_matrix.loc[12350.0].sort_values(ascending=False)
items_bought_by_A = customer_item_matrix.loc[12350.0][customer_item_matrix.loc[12350.0]>0]
print("Items Bought by A: ")
print(items_bought_by_A)


# In[13]:


items_bought_by_B = customer_item_matrix.loc[17935.0][customer_item_matrix.loc[17935.0]>0]
print("Items bought by B:")
print(items_bought_by_B)

print()

items_to_recommend_to_B = set(items_bought_by_A.index) - set(items_bought_by_B.index)
print("Items to Recommend to B ")
print(items_to_recommend_to_B)
df.loc[df['StockCode'].isin(items_to_recommend_to_B),['StockCode', 'Description']].drop_duplicates().set_index('StockCode')


# ## Item-based collaborative filtering

# In[14]:


item_item_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T))
item_item_sim_matrix.columns = customer_item_matrix.T.index

item_item_sim_matrix['StockCode'] = customer_item_matrix.T.index
item_item_sim_matrix = item_item_sim_matrix.set_index('StockCode')


# In[15]:


print(item_item_sim_matrix)


# ### Making Recommendations

# In[16]:


top_10_similar_items = list(item_item_sim_matrix.loc[23166].sort_values(ascending=False).iloc[:10].index)

print(top_10_similar_items)
print()
print(df.loc[
    df['StockCode'].isin(top_10_similar_items),
    ['StockCode', 'Description']
].drop_duplicates().set_index('StockCode').loc[top_10_similar_items])


# In[ ]:




