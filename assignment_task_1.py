import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, datetime
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

dataset = pd.read_csv('ad_org_train.csv')
dataset_test= pd.read_csv('ad_org_test.csv')


def getSeconds(x):
    pat =  re.compile('PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    m = pat.match(str(x))
    return (int(m.group(1))*60*60 if m.group(1) else 0) + (int(m.group(2))*60 if m.group(2) else 0) + (int(m.group(3)) if m.group(3) else 0) if m else 0

dataset['duration'] = dataset['duration'].apply(getSeconds)
dataset_test['duration'] = dataset_test['duration'].apply(getSeconds)


def getYear(a):
     pat =  re.compile('(?:(\d+)-)?(?:(\d+)-)?(?:(\d+)-)?')
     n = pat.match(str(a))
     return (int(n.group(1))if n.group(1) else 0)
dataset['published'] = dataset['published'].apply(getYear)
dataset_test['published'] = dataset_test['published'].apply(getYear)

X=dataset.iloc[:, [2,3,4,5,6,7,8]].values
Y=dataset.iloc[:, 1].values
Z=dataset_test.iloc[:, [1,2,3,4,5,6,7]].values


labelencoder_X = LabelEncoder()
labelencoder_Z = LabelEncoder()
X[:, 6]= labelencoder_X.fit_transform(X[:, 6])
Z[:, 6]= labelencoder_Z.fit_transform(Z[:, 6])

X[X=='F'] = 0
X.astype(int) 

Z[Z=='F'] = 0
Z.astype(int) 

onehotencoder= OneHotEncoder(categorical_features= [6])
X=onehotencoder.fit_transform(X).toarray()
Z=onehotencoder.fit_transform(Z).toarray()


X_train, X_test, Y_train, Y_test= train_test_split(X, Y , test_size=0.5 ,random_state=0)


regressor= RandomForestRegressor(n_estimators=400, max_depth=30,criterion='mse',n_jobs=-1,min_samples_leaf=3,min_samples_split=5)
regressor.fit(X, Y)
regressor.score(X_train, Y_train)

Z_pred= regressor.predict(Z)

Z_pred.to_csv('out.csv', sep=',')