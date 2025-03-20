#!/usr/bin/env python
# coding: utf-8

# #                  *Titanic Passenger Prediction Project*

# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[90]:


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

train_data.head(10)


# In[91]:


train_data.info()
train_data.isnull().sum()


# In[92]:


sns.countplot(x="Survived",data=train_data)
plt.title("the number of survivers and deathes")
plt.show()


# In[93]:


sns.countplot(x="Survived",hue="Sex",data=train_data)
plt.title("The relationship between sex and being survived")
plt.show()


# In[94]:


train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)


# In[95]:


train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

embarked_map = {'S': 0, 'C': 1, 'Q': 2}
train_data['Embarked'] = train_data['Embarked'].map(embarked_map)
test_data['Embarked'] = test_data['Embarked'].map(embarked_map)


# In[96]:


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
x = train_data[features]
y = train_data['Survived']


# In[97]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[98]:


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


# In[99]:


log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train,y_train)


# In[100]:


y_pred = log_model.predict(x_val)
print()
print("the accuracy of model is: ",accuracy_score(y_val,y_pred))


# In[101]:


x_test = test_data[features]
log_preds = log_model.predict(x_test)


# In[102]:


submission = pd.DataFrame({
    "PassengerId" : test_data["PassengerId"],
    'Survived' : log_preds
})
submission.to_csv("C:/Users/m/Desktop/submission_logestic.csv",index=False)


# feature engineering:

# In[103]:


train_data["Title"] = train_data["Name"].str.extract(" ([A-Za-z]+)\." , expand=False)
test_data["Title"] = test_data["Name"].str.extract(" ([A-Za-z]+)\." , expand=False)


# In[104]:


train_data["Family_size"] = train_data['SibSp'] + train_data["Parch"] + 1
test_data["Family_size"] = test_data['SibSp'] + test_data["Parch"] + 1


# In[105]:


train_data["IsAlone"] = (train_data["Family_size"] == 1).astype(int)
test_data["IsAlone"] = (test_data["Family_size"] == 1).astype(int)


# In[106]:


train_data["Has_cabin"] = train_data['Cabin'].notnull().astype(int)
test_data["Has_cabin"] = test_data['Cabin'].notnull().astype(int)


# In[107]:


train_data.head()


# In[108]:


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','Family_size','IsAlone','Has_cabin']
x = train_data[features]
y = train_data['Survived']


# In[109]:


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)


# In[110]:


log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train,y_train)


# In[111]:


y_pred = log_model.predict(x_val)
print("the accuracy of model is: ",accuracy_score(y_val,y_pred))


# In[112]:


x_test = test_data[features]
log_preds = log_model.predict(x_test)


# In[113]:


submission = pd.DataFrame({
    "PassengerId" : test_data["PassengerId"],
    'Survived' : log_preds
})
submission.to_csv("C:/Users/m/Desktop/submission_logestic2.csv",index=False)


# prediction for anyone you wnat: 

# In[128]:


new_passenger = {
    'Pclass': 2,
    'Sex': 1, #male=0,female=1
    'Age': 18,
    'SibSp': 1,
    'Parch': 2,
    'Fare': 7.25,
    'Embarked': 0 #'S' = 0, 'C' = 1, 'Q' = 2
}
new_passenger['Family_size'] = new_passenger['SibSp'] + new_passenger['Parch'] + 1
new_passenger['IsAlone'] = 1 if new_passenger['Family_size'] == 1 else 0
new_passenger['Has_cabin'] = 0

new_passenger_df = pd.DataFrame([new_passenger])
prediction =log_model.predict(new_passenger_df)

if prediction == 1:
    print("will survived")
elif prediction == 0:
    print("will be killed")


# In[ ]:




