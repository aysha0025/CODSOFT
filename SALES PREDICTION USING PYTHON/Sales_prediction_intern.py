#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


# Load the dataset
df = pd.read_csv('advertising.csv')

# Display the first few rows of the dataset
print(df.head())


# In[3]:


# Get the summary statistics
print(df.describe())

# Get information about the dataset
print(df.info())


# In[4]:


# Pairplot to visualize the relationships
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[5]:


# Define the feature (TV) and target (Sales)
X = df[['TV']]
y = df['Sales']


# In[6]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# In[8]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[9]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


# In[12]:


# Ensure X_test is flattened to match y_pred
X_test_flat = X_test.values.flatten()


# In[13]:


# Plot the regression line
plt.scatter(X_test_flat, y_test, color='blue', label='Actual Sales')
plt.plot(X_test_flat, y_pred, color='red', linewidth=2, label='Predicted Sales')
plt.xlabel('TV Advertising Budget (in thousands of dollars)')
plt.ylabel('Sales (in thousands of units)')
plt.title('TV Advertising vs Sales')
plt.legend()
plt.show()


# In[ ]:




