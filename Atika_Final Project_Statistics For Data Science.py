#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')


# In[5]:


import csv, sqlite3

con = sqlite3.connect("boston.db")
cur = con.cursor()


# In[6]:


get_ipython().run_line_magic('reload_ext', 'sql')


# In[7]:


get_ipython().run_line_magic('sql', 'sqlite:///boston.db')


# ## Task 3: Load in the Dataset in your Jupyter Notebook

# In[12]:


import pandas as pd
boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# In[44]:


boston_df.describe()


# In[13]:


boston_df.head()


# ## Task 4: Generate Descriptive Statistics and Visualizations

# In[14]:


#importing important libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 


# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1. For the "Median value of owner-occupied homes" provide a boxplot

# In[16]:


ax = sns.boxplot(y='MEDV', data=boston_df)
plt.title("Median Values of Owner-occupied homes");


# ### 2. Provide a bar plot for the Charles river variable

# In[17]:


sns.catplot(x='CHAS', kind='count', data=boston_df)
plt.xlabel("Charles River Variable");


# ### 3. Provide a boxplot for the MEDV variable vs the AGE variable. (Discretize the age variable into three groups of 35 years and younger, between 35 and 70 years and 70 years and older)

# In[19]:


boston_df.loc[(boston_df.AGE <= 35), 'age_group'] = '35 years and younger'
boston_df.loc[((boston_df.AGE > 35)&(boston_df.AGE < 70)), 'age_group'] = 'Between 35 and 70 years'
boston_df.loc[(boston_df.AGE >= 70), 'age_group'] = '70 years and older'


# In[20]:


ax = sns.boxplot(x='age_group', y='MEDV', data=boston_df)
plt.title("Median value vs Age Groups")
plt.ylabel("Median Value");


# The boxplot above shows that on average the median value of owner occupied homes is higher when the Age is lower

# ### 3. Provide a scatter plot to show the relationship between Nitric oxide concentrations and the proportion of non-retail business acres per town. What can you say about the relationship?

# In[21]:


ax = sns.scatterplot(y='INDUS', x='NOX', data=boston_df)
plt.xlabel("Concentration of Nitric Oxide (in ppm)")
plt.ylabel("Proportion of Non-retail Business Acres");


# From the above scatter plot, it's reflect that there is a positive sloping relationship between concentration of Nitric Oxides and the proportion of non-retail business acres per town. Generally, a higher proprtion of non-retail business acres per town produces a higher concentration of Nitric oxide.

# ### 4.Create a histogram for the pupil to teacher ratio variable

# In[22]:


sns.catplot(y="PTRATIO", kind="count", data=boston_df)
plt.ylabel("Pupil to Teacher Ration");


# In[25]:


sns.histplot(boston_df.PTRATIO, kde=False, bins=15)
plt.xlabel("Pupil to Teacher Ratio");


# ## Task 5: Use the appropriate tests to answer the questions provided.

# ### 1. Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)

# #### Hypothesis
# 
# H
# 0
# :
# µ
# 1
# =
# µ
# 2
#  ("there is no difference in between the median value of houses bounded by Charles river and not bounded.")
# 
# H
# a
# :
# µ
# 1
# ≠
# µ
# 2
#  ("there is a difference in between the median value of houses bounded by Charles river and not bounded.")
# 
# Set α to 0.05

# In[28]:


get_ipython().system('pip install scipy')


# In[31]:


import scipy
boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
boston_df=pd.read_csv(boston_url)


# In[37]:


import scipy as sci


# In[38]:


scipy.stats.ttest_ind(boston_df[boston_df.CHAS == 1].MEDV,
                     boston_df[boston_df.CHAS == 0].MEDV)


# #### The Answer:
# Since the p-value is less than 0.05, we will reject the null hypothesis as there is no significance difference in median value of houses bounded by Charles river and not.

# ### 2. Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)

# #### Hypothesis
# 
# H
# 0
# :
# µ
# 1
# =
# µ
# 2
#  ("There is no difference in median values of houses for occupied units built prior to 1940")
# 
# H
# a
# :
# µ
# 1
# ≠
# µ
# 2
#  ("There is a difference in median values of houses for occupied units built prior to 1940")
# 
# Set α to 0.05

# In[39]:


boston_df.loc[(boston_df.AGE > 81), 'age_span'] = 'before 1940'
boston_df.loc[(boston_df.AGE <= 81), 'age_span'] = 'after 1940'


# In[40]:


scipy.stats.levene(boston_df[boston_df.age_span=='before 1940']['MEDV'],
                  boston_df[boston_df.age_span=='after 1940']['MEDV'],
                  center='mean')


# #### The Answer
# Since the p-value is greater than 0.05, we fail to reject the null hypothesis that there is a statistical difference in median values of houses for each proportion of owner occupied units built prior to 1940.

# ### 3. Can we conclude that there is no relationship between Nitric oxide concentrations and proportion of non-retail business acres per town? (Pearson Correlation)

# #### Hypothesis
# 
# H
# 0
# :
# µ
# 1
# =
# µ
# 2
#  (" There is no relationship between Nitric Oxide concentration and proportion of non-retail business acres per town")
# 
# H
# a
# :
# µ
# 1
# ≠
# µ
# 2
#  ("There is a relationship between Nitric Oxide concentration and proportion of non-retail business acres per town")
# 
# Set α to 0.05

# In[41]:


scipy.stats.pearsonr(boston_df.NOX, boston_df.INDUS)


# #### The Answer
# Since the p-value is greater than 0.05, we can reject the null hypothesis that there is no relationship between the nitric acid concentration and the proportion of non-retail business acres per town.
# 
# And as the r value is positive and close to zero, we can conclude that there is a almost strong relationship between these two variables and the relationship curve will be positively sloping.

# ### 4. •	What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes? (Regression analysis)

# In[42]:


import statsmodels.api as sm


# In[43]:


X = boston_df['DIS']

y = boston_df['MEDV']

X = sm.add_constant(X) 

model = sm.OLS(y, X).fit()
predictions = model.predict(X)

model.summary()


# #### The Answer
# There is an additional impact of 1.0916 of weighted distance to the five Boston employment centres on the median value of owner occupied homes.

# In[ ]:




