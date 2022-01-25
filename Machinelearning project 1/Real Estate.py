#!/usr/bin/env python
# coding: utf-8

# ## Real Estate -Price predictor
# 

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("datafinal.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


housing.hist(bins=50, figsize=(20,15))


# ## Train-Test Splitting

# In[10]:


import numpy as np

def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]    


# In[11]:


#train_set, test_set = split_train_test(housing, 0.2)


# In[12]:


#print(f"Rows in train set:{len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[13]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
print(f"Rows in train set : {len(train_set)}\nRows in test set: {len(test_set)}\n")


# ## Looking for correlation

# In[14]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[15]:


from pandas.plotting import scatter_matrix
attributes = ["RM", "ZN", "MEDV", "LSTAT", "INDUS"]
scatter_matrix(housing[attributes],figsize=(12,8))


# In[16]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# ## Attribute Combination(Trying out)

# In[17]:


housing["TAXRM"]= housing['TAX']/housing['RM']


# In[18]:


housing["TAXRM"]


# In[19]:


housing.head()


# In[20]:


corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[21]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[22]:


housing =train_set.drop("MEDV",axis=1)
housing_labels=train_set["MEDV"].copy()


# ## Creating a Pipeline
# 

# In[23]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])


# In[24]:


housing_num = my_pipeline.fit_transform(housing)


# In[25]:


housing_num #numpyarray


# ## Selecting a model for Real Estate

# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num, housing_labels)


# In[27]:


some_data = housing.iloc[:5]


# In[28]:


some_labels= housing_labels.iloc[:5]


# In[29]:


prepared_data =my_pipeline.transform(some_data)


# In[30]:


model.predict(prepared_data)


# In[31]:


list(some_labels)


# ## Evaluting the model

# In[32]:


from sklearn.metrics import mean_squared_error
housing_prediction = model.predict(housing_num)
mse = mean_squared_error(housing_labels, housing_prediction)
rmse= np.sqrt(mse)


# In[33]:


mse


# ## Using better evalution technique - Cross Validation

# In[34]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num, housing_labels, scoring="neg_mean_squared_error" , cv=10)
rmse_scores = np.sqrt(-scores)


# In[35]:


rmse_scores


# In[36]:


def print_scores(scores):
    print("Scores:", scores)
    print("Mean:",scores.mean())
    print("Standard deviation:", scores.std())


# In[37]:


print_scores(rmse_scores)


# ## Saving the model

# In[38]:


from joblib import dump, load
dump(model, 'Estate.joblib')


# ## Testing the model on test data

# In[39]:


X_test = test_set.drop("MEDV", axis=1)
Y_test = test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
#print(final_predictions, list(Y_test))


# In[40]:


final_rmse


# In[41]:


prepared_data[0]


# ## Using the model

# In[42]:


from joblib import dump, load
import numpy as np
model= load('Estate.joblib')
features = np.array([[ 1.36238896, -0.49297662,  1.02967982, -0.28322059,  1.62488748,
       -0.81983571,  99.13022986, -0.90022106,  1.75306802,  1.60783112,
        0.83272185, -3.87438485,  2.0202982 ]])
model.predict(features)

