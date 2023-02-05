# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 11:56:22 2022

@author: chitr
"""


#importing library
import pandas as pd
import numpy as np
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import cross_val_predict

import warnings
warnings.filterwarnings('ignore')

# data fetching
df_chitra = pd.read_csv("KSI.csv")

#data visualization

df_chitra.head()
df_chitra.shape

df_chitra.values 
df_chitra.columns

df_chitra.dtypes
df_chitra.describe()

#data plotting

correlation=df_chitra.corr()
plt.figure(figsize=(12,9))
sns.heatmap(correlation,annot=True,cmap='ocean')
plt.show()

# class HOUR barplot
sns.barplot(x = 'HOUR', y = 'ACCLASS', data = df_chitra)
plt.show()


# class HOUR barplot
sns.barplot(x = 'LATITUDE', y = 'ACCLASS', data = df_chitra)
plt.show()

# class INJURY count
plt.figure(figsize=(8,4))
sns.countplot(df_chitra['INJURY'])

plt.figure(figsize=(18,4))
sns.countplot(df_chitra['TRAFFCTL'])


plt.figure(figsize=(18,4))
sns.countplot(df_chitra['INVAGE'])

# chequeing unique values per column

print('\nUnique Values per Column')
print(df_chitra.nunique())


# too many unique values to accomodate, so dropping them

uniq_cols=['ObjectId','ACCNUM','DATE','TIME','STREET1','STREET2','NEIGHBOURHOOD','WARDNUM','DIVISION']
for col in df_chitra.columns:
    df_chitra[col].replace('<Null>', np.nan, inplace=True)
    df_chitra[col].replace('unknown', np.nan, inplace=True)
    df_chitra[col].replace('Unknown', np.nan, inplace=True)
    
print('\nMissing Values per Column')
print(df_chitra.isna().sum())

# data cleaning, removing duplicacy, repetation, lot of missing values per column
null_cols=['OFFSET','FATAL_NO','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND','PEDESTRIAN',
            'CYCLIST','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','PASSENGER','SPEEDING','REDLIGHT',
            'ALCOHOL','DISABILITY']

# data cleaning 
df_chitra = df_chitra.drop(null_cols,axis='columns')
df_chitra = df_chitra.drop(uniq_cols,axis='columns')

# deviding them into numeric and categorical column

num_cols=['YEAR','HOUR','LATITUDE','LONGITUDE']
cat_cols=['ROAD_CLASS','DISTRICT','LOCCOORD','ACCLOC','TRAFFCTL','VISIBILITY','LIGHT','RDSFCOND',
          'IMPACTYPE','INVTYPE','INVAGE','INJURY','INITDIR','VEHTYPE','MANOEUVER','DRIVACT',
          'DRIVCOND','AUTOMOBILE','AG_DRIV','POLICE_DIVISION']

# replaceing missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
columns=df_chitra.columns
df_chitra = pd.DataFrame(imputer.fit_transform(df_chitra))
df_chitra.columns = columns

columns

# feature class is x and target class is y
X = df_chitra.drop(['ACCLASS'],axis=1)
y = df_chitra['ACCLASS']


# Datatypes of columns
cat_features=list(X.select_dtypes(include=['object']))

num_features=list(X.select_dtypes(include=['int64','float64']))
print(cat_features)
print(num_features)

cat_f=[]
for i in range(len(df_chitra.columns)):
    if df_chitra.columns[i] in cat_features:
        cat_f.append(i)

y=y.astype('int')
# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=4)

# Preprocessing

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_transformer = Pipeline(steps=[("Scaler",StandardScaler())])
cat_transformer = OneHotEncoder(handle_unknown='ignore')

num_pipeline_chitra = ColumnTransformer(transformers=[
    ('num',num_transformer,num_features),
    ('cat',cat_transformer,cat_features)
    ],remainder='passthrough')


# missing value checking

print(df_chitra.isna().sum())



df_chitra.dtypes

#classifier = SVC(random_state=74)
clf_svm_chitra = SVC(random_state = 74, gamma='auto')

pipe = Pipeline(steps=[
    ('num_pipeline_chitra', num_pipeline_chitra),
    ('svc', clf_svm_chitra) ])
    

pipe.fit(X_train, y_train)

# 5.	Save the accuracy score of the training process to a variable named training_accuracy
y_train_chitra_pred = pipe.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_chitra_pred)
print("Training Accuracy =", training_accuracy)


# 7.	Use the model to predict the test data,  i.e. pass the test features to the model and save the results into a variable named initial_predictions
initial_predictions = pipe.predict(X_test)


# 8.	Calculate the accuracy score and save it to a variable named initial_accuracy and print it out.
initial_accuracy = accuracy_score(y_test, initial_predictions)
print("Initial Accuracy =", initial_accuracy)

# 4) Fine tune the model
from sklearn.model_selection import GridSearchCV
param_grid= {'svc__kernel': ['linear', 'rbf'],
            'svc__C': [0.01, 0.1, 1],
            'svc__gamma': [0.01, 0.06, 0.1]
        }


clf_svm_chitra = GridSearchCV(pipe,param_grid,refit=True,verbose=3)
clf_svm_chitra.fit(X_train, y_train)
print(clf_svm_chitra.best_params_)



best_model_chitra = clf_svm_chitra.best_estimator_


final_predictions = best_model_chitra.predict(X_test)

final_accuracy = accuracy_score(y_test, final_predictions)
print("Final Accuracy =", final_accuracy)





#accuracy scores

from sklearn.metrics import confusion_matrix
prediction_score = confusion_matrix(y_test,final_predictions)
print(prediction_score)


