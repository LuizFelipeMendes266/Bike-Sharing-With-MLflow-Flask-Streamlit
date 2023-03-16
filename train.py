# -*- coding: utf-8 -*-
import seaborn as sns
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler


"""Loading"""

train=pd.read_csv('data/train_bikes.csv')
test=pd.read_csv('data/test_bikes.csv')

"""Shape train and Test"""

train.shape , test.shape

"""Summary Statistics"""

train.describe()

"""We don't have nulls values"""

train.isnull().sum()

"""In this case, let's add some columns provided by the datetime. We can drop 'datetime' after that."""

train['datetime'] = pd.to_datetime(train['datetime'])
train['hour'] = train['datetime'].dt.hour
train['month'] = train['datetime'].dt.month
train['weekday'] = train['datetime'].dt.weekday
train.drop(columns=['datetime'] ,inplace=True)
train

"""Same thing to test Dataset."""

test['datetime'] = pd.to_datetime(test['datetime'])
test['hour'] = test['datetime'].dt.hour
test['month'] = test['datetime'].dt.month
test['weekday'] = test['datetime'].dt.weekday
test.drop(columns=['datetime'] ,inplace=True)
test

"""Let's plot a pairplot with the histogram type diagonal."""

sns.pairplot(train,diag_kind='hist')

"""Histogram of our target feature."""

sns.displot(train['count'])

"""So, we do not have a distribution that we're searching for. In this case, we can transform with log and see what happens."""

sns.displot(np.log(train['count']))

"""It's better with a log transformation. Let's apply that. Continuing with the exploration, we can graphically visualize how the count occurs by grouping by day of the week and hour."""

mean_hour=train.groupby(['hour','weekday'])['count'].mean().reset_index()
mean_hour

"""Let's plot that."""

plt.figure(figsize=(25,10))
sns.barplot(x='weekday',y='count',data=mean_hour,hue='hour')

"""We can see that on workdays, we have higher occurrences at 5 pm. On weekends, around 1 pm.

In relation to the month, we have higher occurrences in 6, that is, June.
"""

sns.barplot(x='month',y='count',data=train)

"""In relation to the season, we have higher occurrences in 3, that is, autumn."""

sns.barplot(x='season',y='count',data=train)

"""Let's take a look at the correlation plot."""

plt.figure(figsize=(20,20))
sns.heatmap(train.corr(),annot=True)

"""# Feature Emgineering

I would like to build a column that shows us whether the occurrence is during rush hour or not, categorized as 0 or 1. Let's create a function for that.
"""

def rush_hour(X):
  df=X.copy()
  c1=((df['weekday'].isin([0,6])) & ((df['hour']>=10) & (df['hour']<=17)))
  c2=((df['weekday'].isin([0,6])==False) & (((df['hour']>=17) & (df['hour']<=19)) | (df['hour']==8)))
  df['peak']=np.where(c1,1,np.where(c2,1,0))
  return df

"""This is a function called rush_hour that takes in a pandas DataFrame X as input and returns a modified copy of it.

The function first creates two boolean conditions c1 and c2 based on the values in the weekday and hour columns of the DataFrame. These conditions are used to identify rush hours, which are typically times of the day when traffic is heaviest.

c1 represents rush hours during weekends (i.e., weekdays 0 and 6, which correspond to Sunday and Saturday, respectively) from 10:00 AM to 5:00 PM.

c2 represents rush hours during weekdays (i.e., weekdays 1-5) from 5:00 PM to 7:00 PM, and also from 8:00 AM to 9:00 AM (which is often considered a peak hour for traffic).

The function then creates a new column called pico in the DataFrame using the np.where() function. If c1 or c2 is true for a particular row, the corresponding value of pico is set to 1. Otherwise, pico is set to 0.

# Modeling

We split the data in training and test
"""

from sklearn.model_selection import train_test_split

X = train[['season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed', 'hour', 'month', 'weekday']]
y = train[['count']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train

# reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )
# models,predictions = reg.fit(X_train, X_test, y_train, y_test)

"""## Pipeline

So, as we discussed before, we can transform the target with log. To do that, let's produce a function to that and use it in the pipeline.
"""

def log_target(target):
  return np.log(target)

def exp_log(target):
  return np.exp(target)

pipeline=Pipeline([
          ('transformer_pico', FunctionTransformer(rush_hour)),
          ('target_transform_plus_regressor',TransformedTargetRegressor(regressor=RandomForestRegressor(),func=log_target,inverse_func=exp_log))
])

"""This is a machine learning pipeline that applies a custom function to the input data, standardizes the data, and then uses a Random Forest Regressor to make predictions on a transformed target variable."""

pipeline.fit(X_train,y_train)

y_pred_train=pipeline.predict(X_train)
y_pred_test=pipeline.predict(X_test)

print(f' r2 test score: {r2_score(y_test,y_pred_test)}')
print(f' r2 train score: {r2_score(y_train,y_pred_train)}')

print(f' Test MRE: {mean_squared_error(y_test,y_pred_test)}')
print(f' Train MRE: {mean_squared_error(y_train,y_pred_train)}')

"""Cross Validation Value

The bike share demand dataset is a time series problem because it involves predicting the number of bikes that will be rented at a given time, which can vary depending on the day of the week, season, weather conditions, and other factors. To evaluate the performance of a machine learning model for this problem, it is important to use a time series cross-validation approach, such as the TimeSeriesSplit method with n_splits=5, to ensure that the model is not overfitting to specific time periods in the dataset.
"""

cv=TimeSeriesSplit(n_splits=5)
score=cross_val_score(pipeline,X_train,y_train,scoring='r2',cv=cv)
score

"""In the provided code, a pipeline is used to preprocess the data and fit a regression model to predict the bike rental count. The cross_val_score function is used to calculate the R-squared score for each fold of the time series cross-validation, which measures the proportion of variance in the target variable (i.e., bike rental count) that is explained by the model. The resulting scores can be used to evaluate the overall performance of the model and compare different models or hyperparameters."""

score.mean()

"""# MLflow Metrics

In this part, we will save graphs, evaluation metrics, and the model using MLflow.
"""

import mlflow
from mlflow.models.signature import infer_signature

"""Set and run the mlflow experiment"""

mlflow.set_experiment('Prediction Bike')
mlflow.start_run()

"""Saving the first Graphic"""

plt.figure(figsize=(25,10))
sns.barplot(x='weekday',y='count',data=mean_hour,hue='hour')
plt.savefig("mlruns/graph_hours.png")
mlflow.log_artifact("mlruns/graph_hours.png")

plt.close()

"""Second Graphic"""

sns.displot(np.log(train['count']))
plt.savefig("mlruns/target.png")
mlflow.log_artifact("mlruns/target.png")

plt.close()

"""Loading metrics"""

r2 = r2_score(y_test,y_pred_test)
mre = mean_squared_error(y_test,y_pred_test)
cv=TimeSeriesSplit(n_splits=5)
score = cross_val_score(pipeline,X_train,y_train,scoring='r2',cv=cv)
score = score.mean()

mlflow.log_metric('r2 test', r2)
mlflow.log_metric('Mean Squared Error',mre)
mlflow.log_metric('Cross Validation Mean',score)

"""Model"""

mlflow.sklearn.log_model(pipeline,"model_pipeline")

"""Loading Params"""

params = pipeline.named_steps['target_transform_plus_regressor'].get_params()

mlflow.log_params(params)

"""Finishing """

mlflow.end_run()

mlflow.search_runs()

"""As we always want to work with the latest version of the model, we created an artifact for this."""

# last_run = dict(mlflow.search_runs().sort_values(by='start_time',ascending=False).iloc[0])
# artifact = last_run['artifact_uri']

# """Now, let's try load the latest version of the model provided by MLflow."""

# model = mlflow.sklearn.load_model(artifact+"/model_pipeline")

# """Let's try"""

# model

# model.predict(X_test)

"""As we can see, we loaded the model perfectly.

# Flask APP with the test Data

"""

# test

# """Selecting the first row to test our Flask App with this data."""

# test.iloc[0]

# import json

# data = test.iloc[0].to_dict()
# json_string = json.dumps(data)

# # Escrever string JSON em um arquivo
# with open('data.json', 'w') as f:
#     f.write(json_string)