import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_Y = train_data['Transported']
train_X = train_data.drop(['Transported'], axis=1)

train_X = train_X.drop(['PassengerId', 'Name', 'Destination'], axis=1)
test_data = test_data.drop(['PassengerId', 'Name', 'Destination'], axis=1)

train_X = train_X.dropna()
test_X = test_data.dropna()


train_X['HomePlanet'] = train_X['HomePlanet'].map({'Europa':1, 'Earth':0})
test_X['HomePlanet'] = test_X['HomePlanet'].map({'Europa':1, 'Earth':0})

# train_X['VIP'] = train_X['VIP'].map({'True':1, 'False':0})
# test_X['VIP'] = test_X['VIP'].map({'True':1, 'False':0})


encoder = LabelEncoder()
train_X['VIP'] = encoder.fit_transform(train_X['VIP'])
test_X['VIP'] = encoder.transform(test_X['VIP'])


encoder = LabelEncoder()
train_X['CryoSleep'] = encoder.fit_transform(train_X['CryoSleep'])
test_X['CryoSleep'] = encoder.transform(test_X['CryoSleep'])


encoder = LabelEncoder()
train_X['Cabin'] = encoder.fit_transform(train_X['Cabin'])
test_X['Cabin'] = encoder.transform(test_X['Cabin'])

train_X