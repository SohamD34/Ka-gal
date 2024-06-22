import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse 

train_df = pd.read_csv('/kaggle/input/playground-series-s4e5/train.csv')

test_X = pd.read_csv('/kaggle/input/playground-series-s4e5/test.csv')


train_df = train_df.drop(['id'], axis=1)
test_X_ids = test_X['id']
test_X = test_X.drop(['id'], axis=1)

corr_map = sns.heatmap(train_df.corr())
plt.show()


## All features are independent of each other - no need to find principle components
## Checking correlation of each feature with FloodProbability

features = list(test_X.columns)
d = {}
for i in features:
    d[i] =  np.corrcoef(train_df[i], train_df['FloodProbability'])[0][1]
d = dict(sorted(d.items(), key=lambda item: item[1]))
print(d)


X_train = train_df[features].drop(['CoastalVulnerability'], axis=1)
Y_train = train_df['FloodProbability']

test_X = test_X.drop(['CoastalVulnerability'], axis=1)


## Prediction

model = CatBoostRegressor(loss_function = 'RMSE')
parameters ={'learning_rate':[0.001, 0.005, 0.01, 0.05]}
predictor = GridSearchCV(model, parameters)

predictor.fit(X_train, Y_train, verbose=10000)
print(predictor.best_params_ )


# Generate predictions on the training and validation sets using the trained 'model' 
y_train = model.predict(X_train) 
      
# Calculate and print the Root Mean Squared Error (RMSE) for training and validation sets 
print("Training RMSE: ", np.sqrt(mse(Y_train, y_train))) 

y_pred = model.predict(test_X)
df = pd.DataFrame()
df['id'] = test_X_ids
df['FloodProbability'] = y_pred

df.to_csv('catboost.csv',index=False)