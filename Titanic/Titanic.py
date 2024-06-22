import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.stats.stats import pearsonr
from sklearn.model_selection import GridSearchCV

train_data = pd.read_csv('train.csv')
test_X = pd.read_csv('test.csv')

train_X = train_data.drop('Survived', axis=1)
train_Y = train_data['Survived']

test_X_copy = test_X.copy()
train_X.head(5)

print(train_X.isnull().sum())
print("Total rows = ", len(train_X))


## Data Preprocessing

train_X = train_X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_X = test_X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

train_X['Age'] = train_X['Age'].fillna(train_X['Age'].mean())
test_X['Age'] = test_X['Age'].fillna(test_X['Age'].mean())

train_X['Embarked'] = train_X['Embarked'].fillna(train_X['Embarked'].mode()[0])
test_X['Embarked'] = test_X['Embarked'].fillna(test_X['Embarked'].mode()[0])

train_X['Fare'] = train_X['Fare'].fillna(train_X['Fare'].mean())
test_X['Fare'] = test_X['Fare'].fillna(test_X['Fare'].mean())

train_X['Sex'] = train_X['Sex'].map({'male':0, 'female':1})
test_X['Sex'] = test_X['Sex'].map({'male':0, 'female':1})

train_X['Embarked'] = train_X['Embarked'].map({'S':0, 'C':1, 'Q':2})
test_X['Embarked'] = test_X['Embarked'].map({'S':0, 'C':1, 'Q':2})

train_X[['Age']] = train_X[['Age']]/100
test_X[['Age']] = test_X[['Age']]/100

scaler = MinMaxScaler()
train_X[['Fare']] = scaler.fit_transform(train_X[['Fare']])
test_X[['Fare']] = scaler.transform(test_X[['Fare']])

train_X.head(5)

for cols in train_X.columns:
    print(cols)
    print(pearsonr(train_X[cols], train_Y))



## Model Building


train_X = train_X.drop(['SibSp', 'Parch'], axis=1)
test_X = test_X.drop(['SibSp', 'Parch'], axis=1)

dtc = sklearn.ensemble.RandomForestClassifier()
grid = {'n_estimators':[2,3,4,5,6], 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]}

gscv = GridSearchCV(dtc, grid, cv=5)
gscv.fit(train_X, train_Y)

print(gscv.best_params_)
print(gscv.best_score_)
print(gscv.score(train_X, train_Y))

train_pred = gscv.predict(train_X)
print("Score =",sklearn.metrics.accuracy_score(train_pred, train_Y))


# Submission

predicted = gscv.predict(test_X)
psg_id = test_X_copy['PassengerId']

predictions = pd.DataFrame()
predictions['PassengerId'] = psg_id
predictions['Survived'] = predicted

predictions.head(5)
predictions.to_csv('dtc.csv', index=False)