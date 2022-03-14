#!/usr/bin/env python
# coding: utf-8

# # GRIP - THE SPARK FOUNDATION
# 
# ### DATA SCIENCE & BUSINESS ANALYTICS INTERNSHIP
# 
# ## NIRMAL JOY

# ## TASK 1 - : PREDICTION USING SUPERVISED ML

# 
# ### PREDICT THE PERCENTAGE OF AN STUDENT BASED ON THE NO. OF STUDY HOURS
# 

# ### 1. IMPORT REQUIRED LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ### 2. IMPORTING DATASET

# In[2]:


Score = pd.read_csv(r'C:\Users\vpare\OneDrive\Desktop\Virtual internship\GRIP\TS1\Score.csv')
print("Dataset imported...")


# In[3]:


Score.head()


# ## 3. DATA VISUALIZATION

# ### 3.1 PLOTTING THE DISTRIBUTION HOURS AND SCORES ON SCATTER PLOTS

# In[4]:


plt.scatter(x="Hours",y="Scores",data=Score)
plt.title("Hours vs Percentage")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")


# ### 3.2 UNIVARIATE VISUALIZATION

# In[5]:


Score.hist(grid=False,figsize=(8,3))
plt.show()


# ## 4. DATA PREPERATION

# In[6]:


x =Score.drop(columns=["Scores"])
y = Score["Scores"]


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)
print(f"--> training data points {x_train.shape[0]} and testing data points {x_test.shape[0]}")


# ## 5. TRAINING THE ALGORITHM

# In[8]:


regressor = LinearRegression()
regressor.fit(x_train,y_train)
print("training is Completed ")


# ### 5.1 PLOTTING REGRESSION LINE

# In[9]:


print(f"--> Equation of line y = {np.round(regressor.coef_[0],2)}*x+{np.round(regressor.intercept_,2)}")


# In[10]:


plt.scatter(x_train,y_train)
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.plot(x_train,regressor.predict(x_train),"m")
plt.show()


# ## 6. MAKING PREDICTIONS ON TESTING DATA

# In[11]:


print(x_test) #testing data


# In[12]:


y_pred = regressor.predict(x_test)


# In[13]:


print("Comparing actual data and  predicted data")
df = pd.DataFrame({"Actual":y_test, "Predicted": y_pred})
df


# ### 6.1 PLOTTING LINE ON TESTING DATA

# In[14]:


plt.scatter(x_test,y_test)
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.plot(x_test,regressor.predict(x_test),"g")
plt.show()


# ## 7. TESTING AND FINDING PREDICTION FOR 9.25 HOURS STUDY/DAY

# In[15]:


hours = 9.25
own_pred = regressor.predict([[hours]])
print(f"No of Hours = {hours}")
print(f"predicted Score = {own_pred[0]}")


# ## 8. EVALUATING MODEL

# In[16]:


print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred))
print("Mean squared Error:",metrics.mean_squared_error(y_test,y_pred))
print("r2 Score:",metrics.r2_score(y_test,y_pred))


# # Thank you ❤❤
