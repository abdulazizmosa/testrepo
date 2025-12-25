#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install numpy pandas matplotlib seaborn scikit-learn


# #Import Required Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


df = sns.load_dataset('titanic')
df.head()


# Check dataset info:

# In[3]:


df.info()


# #Select Features and Target
# #Target Variable

# In[6]:


y = df['survived']


# #Drop Irrelevant Features

# In[7]:


features_to_drop = [
    'survived', 'alive', 'deck', 'embark_town', 
    'class', 'who', 'adult_male'
]

X = df.drop(columns=features_to_drop)


# #5️⃣ Exercise 1: How Balanced Are the Classes?

# In[8]:


y.value_counts()


# #6️⃣ Train-Test Split (With Stratification)

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# #7️⃣ Separate Numerical & Categorical Features

# In[10]:


numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()


# #8️⃣ Preprocessing Pipelines

# #Numerical Pipeline
# 
# #Fill missing values with median
# 
# #Standardize values

# In[11]:


num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# #Categorical Pipeline
# #Fill missing values
# #One-Hot Encode

# In[12]:


cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# #9️⃣ Combine Pipelines (ColumnTransformer)

# In[13]:


preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numerical_features),
    ('cat', cat_pipeline, categorical_features)
])


# #🔟 Create the Full Model Pipeline (Random Forest)

# In[14]:


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# #1️⃣1️⃣ Define Parameter Grid (Random Forest)

# In[15]:


param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__class_weight': [None, 'balanced']
}


# #1️⃣2️⃣ Grid Search with Cross-Validation

# In[16]:


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2
)

model.fit(X_train, y_train)


# #1️⃣3️⃣ Exercise 4: Predict on Test Data

# In[17]:


y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# #1️⃣4️⃣ Exercise 5: Confusion Matrix (Random Forest)

# In[18]:


conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Titanic Classification Confusion Matrix (Random Forest)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# #1️⃣5️⃣ Feature Importance (Random Forest)

# In[19]:


importances = model.best_estimator_.named_steps['classifier'].feature_importances_


# #1️⃣6️⃣ Try Another Model: Logistic Regression

# In[20]:


param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

model.param_grid = param_grid


# #New Parameter Grid

# In[25]:


param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

model.param_grid = param_grid


# #1️⃣7️⃣ Train Logistic Regression Model

# In[26]:


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))


# In[27]:


'classifier__penalty'


# In[28]:


logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])


# In[29]:


logreg_param_grid = {
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}


# In[30]:


logreg_model = GridSearchCV(
    estimator=logreg_pipeline,
    param_grid=logreg_param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2
)


# In[31]:


logreg_model = GridSearchCV(
    estimator=logreg_pipeline,
    param_grid=logreg_param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=2
)


# In[32]:


logreg_model.fit(X_train, y_train)


# In[33]:


y_pred = logreg_model.predict(X_test)

print(classification_report(y_test, y_pred))


# #1️⃣8️⃣ Confusion Matrix (Logistic Regression)

# In[34]:


conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Titanic Classification Confusion Matrix (Logistic Regression)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# #1️⃣9️⃣ Extract Logistic Regression Coefficients

# In[35]:


coefficients = (
    logreg_model
    .best_estimator_
    .named_steps['classifier']
    .coef_[0]
)


# In[ ]:


#Get Feature Names (IMPORTANT STEP)


# In[36]:


categorical_feature_names = (
    logreg_model
    .best_estimator_
    .named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(categorical_features)
)

feature_names = numerical_features + list(categorical_feature_names)


# #2️⃣0️⃣ Plot Feature Coefficient Magnitudes

# In[37]:


importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Coefficient'].abs())
plt.gca().invert_yaxis()
plt.title('Feature Coefficient Magnitudes (Logistic Regression)')
plt.xlabel('Coefficient Magnitude')
plt.show()


# In[ ]:


#2️⃣1️⃣ Final Test Accuracy


# In[38]:


test_score = logreg_model.best_estimator_.score(X_test, y_test)
print(f"Test set accuracy: {test_score:.2%}")


# In[ ]:




