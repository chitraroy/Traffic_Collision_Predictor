from flask import Flask, request, jsonify, render_template
import traceback
import pandas as pd
import joblib
import sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import metrics

models = {'RandomForest':'Kanishka_clf.pkl',
          'SVC':'Chitra_clf.pkl',
          'NeuralNet':'Vikas_clf.pkl',
          }

uniq_cols=['ObjectId','ACCNUM','DATE','TIME','STREET1','STREET2','NEIGHBOURHOOD','WARDNUM','DIVISION']
null_cols=['OFFSET','FATAL_NO','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND','PEDESTRIAN',
            'CYCLIST','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','PASSENGER','SPEEDING','REDLIGHT',
            'ALCOHOL','DISABILITY']

rawdata = pd.read_csv('KSI.csv',index_col=('INDEX_'))
for col in rawdata.columns:
    rawdata[col].replace('<Null>', np.nan, inplace=True)
    rawdata[col].replace('unknown', np.nan, inplace=True)
    rawdata[col].replace('Unknown', np.nan, inplace=True)
rawdata = rawdata.drop(null_cols,axis='columns')
rawdata = rawdata.drop(uniq_cols,axis='columns')

rawdata['ACCTYPE'] = np.where(rawdata['ACCLASS'] == 'Fatal', 1, 0)
imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
columns=rawdata.columns
rawdata = pd.DataFrame(imputer.fit_transform(rawdata))
rawdata.columns = columns

X = rawdata.drop(['ACCLASS','ACCTYPE'],axis=1)
y = rawdata['ACCTYPE']
# Datatypes of columns
cat_features=list(rawdata.select_dtypes(include=['object']))
int_features=list(rawdata.select_dtypes(include=['int64']))
float_features=list(rawdata.select_dtypes(include=['float64']))
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=4)


# Your API definition
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/scores/<model_name>", methods=['GET','POST']) #use decorator pattern for the route
def scores(model_name):
    if loaded_model:
        try:
            y_pred = loaded_model[model_name].predict(X_test)
            print(f'Returning scores for {model_name}:')
            
            accuracy = metrics.accuracy_score(y_test, y_pred)
            precision = metrics.precision_score(y_test, y_pred)
            recall = metrics.recall_score(y_test, y_pred)
            f1 = metrics.f1_score(y_test, y_pred)
            print(f'accuracy={accuracy}  precision={precision}  recall={recall}  f1={f1}')
            res = jsonify({"accuracy": accuracy,
                            "precision": precision,
                            "recall":recall,
                            "f1": f1,
                            'Columns': list(X.columns),
                            'Values': list(X_test.iloc[12])
                           })
            res.headers.add('Access-Control-Allow-Origin', '*')
            return res
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model available.')

@app.route("/predict/<model_name>", methods=['GET','POST']) #use decorator pattern for the route
def predict(model_name):
    if loaded_model:
        try:
            data = request.get_json()
            print('JSON: \n', data)
            values = data["values"]
            columns = data["columns"]
            df = pd.DataFrame(data=[values], columns=columns)
            print(df)
            
            for col in df.columns:
                df[col].replace('<Null>', np.nan, inplace=True)
                df[col].replace('unknown', np.nan, inplace=True)
                df[col].replace('Unknown', np.nan, inplace=True)
            
            df = pd.DataFrame(imputer.fit_transform(df))
            df.columns = columns
            
            df[int_features]=df[int_features].astype('int')
            df[float_features]=df[float_features].astype('float')

            prediction = list(loaded_model[model_name].predict(df))
            print(f'Returning prediction with {model_name} model:')
            print('prediction=', prediction)
            print(df.dtypes)
            print('Y Test',y_test.iloc[12])
            if prediction[0] == 0:
                res = jsonify({"prediction": "Non-Fatal"})
            elif prediction[0] == 1:
                res = jsonify({"prediction": "Fatal"})
            res.headers.add('Access-Control-Allow-Origin', '*')
            return res
        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model available.')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
        
    loaded_model={}
    for model_name in (models):
        loaded_model[model_name] = joblib.load(models[model_name])
        print(f'Model {model_name} loaded')
        
    
    app.run(port=port, debug=True)
