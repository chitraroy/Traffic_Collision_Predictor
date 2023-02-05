import pandas as pd
import numpy as np


rawdata = pd.read_csv('C:/Users/Kanishka_Dhir/Documents/Sem-2/COMP 247-Supervised Learning/Project/KSI.csv',index_col=('INDEX_'))

# Some columns have too many unique values and hence will have too high varience to be useful
uniq_cols=['ObjectId','ACCNUM','DATE','TIME','STREET1','STREET2','NEIGHBOURHOOD','WARDNUM','DIVISION']

for col in rawdata.columns:
    rawdata[col].replace('<Null>', np.nan, inplace=True)
    rawdata[col].replace('unknown', np.nan, inplace=True)
    rawdata[col].replace('Unknown', np.nan, inplace=True)

print('\nMissing Values per Column')
print(rawdata.isna().sum())

# Many columns have too much data missing hence we cannot use these columns
null_cols=['OFFSET','FATAL_NO','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND','PEDESTRIAN',
            'CYCLIST','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','PASSENGER','SPEEDING','REDLIGHT',
            'ALCOHOL','DISABILITY']

# Removing unusable columns
rawdata = rawdata.drop(null_cols,axis='columns')
rawdata = rawdata.drop(uniq_cols,axis='columns')

print(rawdata.dtypes)
# Imputer for missing entries
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
columns=rawdata.columns
rawdata = pd.DataFrame(imputer.fit_transform(rawdata))
rawdata.columns = columns

X = rawdata.drop(['ACCLASS'],axis=1)
y = rawdata['ACCLASS']

# Datatypes of columns
cat_features=list(X.select_dtypes(include=['object']))
num_features=list(X.select_dtypes(include=['int64','float64']))
print(cat_features)
print(num_features)

cat_f=[]
for i in range(len(rawdata.columns)):
    if rawdata.columns[i] in cat_features:
        cat_f.append(i)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=4)
#y_train=y_train.astype('int')
#y_test=y_test.astype('int')
# Preprocessing

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_transformer = Pipeline(steps=[("Scaler",StandardScaler())])
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num',num_transformer,num_features),
    ('cat',cat_transformer,cat_features)
    ],remainder='passthrough')

# Classification
from sklearn.ensemble import RandomForestClassifier
classifier_kanishka = RandomForestClassifier(max_depth=5,random_state=57)

# Pipeline

pipe = Pipeline([
    ('preprocessor',preprocessor),
    ('clf',classifier_kanishka)
    ])

print(pipe.steps)

pipe.fit(X_train,y_train)
print("model score: %.3f" % pipe.score(X_test, y_test))

from sklearn import metrics
y_pred = pipe.predict(X_train)
training_accuracy = metrics.accuracy_score(y_train, y_pred)
print("Initial Accuracy =", training_accuracy)

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(1, 50)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'clf__n_estimators': n_estimators,
               'clf__max_features': max_features,
               'clf__max_depth': max_depth,
               'clf__min_samples_split': min_samples_split,
               'clf__min_samples_leaf': min_samples_leaf,
               'clf__bootstrap': bootstrap}
#print(random_grid)

rf_random = RandomizedSearchCV(estimator = pipe, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=57, n_jobs = -1)
rf_random.fit(X_train,y_train)
print(rf_random.best_params_)
print(rf_random.best_estimator_)

from sklearn import metrics
best_randomclaasifier_kanishka = rf_random.best_estimator_
y_test_pred = best_randomclaasifier_kanishka.predict(X_test)
randomAccuracyScore = metrics.accuracy_score(y_test,y_test_pred)
print("accuracy of model on best random forest",randomAccuracyScore)


# import joblib
# #joblib.dump(pipe,'Kanishka_clf.pkl')


# model = joblib.load('Kanishka_clf.pkl')
# p = model.predict(X_test)
