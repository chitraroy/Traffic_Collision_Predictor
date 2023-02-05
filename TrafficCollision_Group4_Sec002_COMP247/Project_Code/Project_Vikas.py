import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

rawdata = pd.read_csv('KSI.csv',index_col=('INDEX_'))

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

rawdata['ACCTYPE'] = np.where(rawdata['ACCLASS'] == 'Fatal', 1, 0)

# Imputer for missing entries
imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
columns=rawdata.columns
rawdata = pd.DataFrame(imputer.fit_transform(rawdata))
rawdata.columns = columns

X = rawdata.drop(['ACCLASS','ACCTYPE'],axis=1)
y = rawdata['ACCTYPE']

y = y.astype('int')

# Datatypes of columns
cat_features=list(X.select_dtypes(include=['object']))
num_features=list(X.select_dtypes(include=['int64','float64']))

num_transformer = Pipeline(steps=[
    ('scaler',StandardScaler())])

cat_transformer = Pipeline(steps=[
    ('encoder',OneHotEncoder(handle_unknown='ignore',sparse=False))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
      ],remainder='passthrough')

classifier = MLPClassifier(hidden_layer_sizes=(20,18,16,5),
                           max_iter=100,activation = 'tanh',
                           solver='adam',random_state=4)

pipe = Pipeline(steps=[('pre',preprocessor),('clf', classifier)])
pipe.fit(X_train,y_train)
print(pipe.score(X_test,y_test))

import joblib
joblib.dump(pipe,'Vikas_clf.pkl')

