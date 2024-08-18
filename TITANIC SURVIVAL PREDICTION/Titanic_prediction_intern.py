#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV


# In[2]:


df = pd.read_csv('Titanic-Dataset.csv')


# In[3]:


print(df.head())


# In[4]:


print(df.info())
print(df.describe())


# In[5]:


print(df.isnull().sum())


# In[6]:


# Fill missing values in the 'Age' column with the median value
df['Age'].fillna(df['Age'].median(), inplace=True)


# In[8]:


# Fill missing values in the 'Embarked' column with the most frequent value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


# In[9]:


# Drop the 'Cabin' column as it has too many missing values
df.drop(columns=['Cabin'], inplace=True)


# In[10]:


# Verify that there are no more missing values
print(df.isnull().sum())


# In[11]:


# Feature Engineering: Convert 'Sex' column to numerical values
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})


# In[12]:


# One-hot encode the 'Embarked' column
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# In[13]:


# Drop unnecessary columns
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)


# In[14]:


# Display the updated DataFrame
print(df.head())


# In[15]:


# Define features (X) and target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']


# In[16]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# In[18]:


# Make predictions on the test set
y_pred = model.predict(X_test)


# In[19]:


# Evaluate the Logistic Regression model
print("Logistic Regression Model Evaluation")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))


# In[20]:


# Hyperparameter Tuning and Model Comparison using RandomForestClassifier
# Define a parameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[21]:


# Initialize a GridSearchCV object
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)


# In[22]:


# Fit the grid search to the data
grid_search.fit(X_train, y_train)


# In[23]:


# Print the best parameters and the best score
print("Best Parameters from GridSearchCV")
print(grid_search.best_params_)
print("Best Score from GridSearchCV")
print(grid_search.best_score_)


# In[24]:


# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)


# In[25]:


# Evaluate the RandomForestClassifier model
print("RandomForestClassifier Model Evaluation")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))


# In[ ]:




