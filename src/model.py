#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lightgbm as lgb
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
import gc


# In[2]:


import time

# Function to make presiction and save it to csv file in kaggle submission format
def save_submission(model, extension="", transform=None):
    
    if transform is None:
        # Converting predictions into exp(x) - 1
        pred = np.expm1(model.predict(test.drop(columns=["timestamp","row_id"])))
    else:
        pred = np.expm1(model.predict(transform(test.drop(columns=["timestamp","row_id"]))))
        
    filename = time.strftime("%y-%m-%d-%H%M%S") + extension +".csv"

    # Saving predictions to csv file
    pd.DataFrame(pred, columns=["meter_reading"]).rename_axis("row_id").to_csv(filename)
    
    #converts the csv file to 7z to reduce size
    get_ipython().system('p7zip $filename >> /dev/null')
    print("{} saved!".format(filename))

# save model to disk in pickle format
def save_model(model, name="model.pkl"):
    with open("../models/" + name, "wb") as f:
        pickle.dump(model, f)
    print("{} saved to disk!".format(name))


# In[3]:


with open("../data/train_featurized.pkl", "rb") as f:
    train = pickle.load(f)


# In[4]:


train


# In[5]:


X = train.drop(columns=["timestamp","meter_reading"])

# converting meter_reading to y where y=log(meter_reading + 1)
y = np.log1p(train["meter_reading"])


# In[6]:


from sklearn.metrics import make_scorer

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log(y_pred+1) - np.log(y_true+1))))

rmsle_error = make_scorer(rmsle, greater_is_better=False)


# ### Lgbm Model

# In[12]:


param_grid = {
    "learning_rate" : [0.01],
    "n_estimators" : [512, 1024],
    "num_leaves" : [32,64],
    "objective" : ["regression"],
    "metric" : ["rmse"],
    "colsample_bytree": [0.9],
}

lgbm = LGBMRegressor()
grid = GridSearchCV(lgbm, param_grid, scoring="neg_mean_squared_error", cv=3, return_train_score=True)
grid.fit(X, y)


# In[13]:


grid.cv_results_


# In[14]:


grid.best_params_


# In[16]:


features = ['building_id', 'meter', 'DT_M', 'DT_W', 'DT_D', 'DT_hour',
       'DT_day_week', 'DT_day_month', 'DT_week_month', 'DT_h_sin', 'DT_h_cos',
       'weekend', 'site_id', 'primary_use', 'square_feet', 'year_built',
       'air_temperature', 'cloud_coverage', 'dew_temperature',
       'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
       'wind_speed']

categorical_features = ['building_id', 'meter', 'DT_M', 'DT_W',
       'DT_D', 'DT_hour', 'DT_day_week', 'DT_day_month', 'DT_week_month', 'weekend',
       'site_id', 'primary_use', 'year_built']

train_data = lgb.Dataset(X,label=y, categorical_feature=categorical_features, feature_name=features)


# In[17]:


param= {
    "learning_rate" : 0.01,
    "num_iterations" : 1024,
    "num_leaves" : 64,
    "objective" : "regression",
    "metric" : "rmse",
    "colsample_bytree": 0.9,

}

lgbm = lgb.train(param, train_data, categorical_feature=categorical_features, feature_name=features)

save_submission(lgbm, "lgbm")
save_model(lgbm,"lgbm.pkl")


# <img src="../images/model/kaggle/lgbm.png">

# ### Imputation with median

# In[7]:


X.isnull().sum()


# In[8]:


# all these are numerical features
features_to_impute = [
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed"
]

impute_vals = {}

# Median imputation

for site_id in range(0,15+1):
    site = {}
    for feature in features_to_impute:
        median = X[X["site_id"]==site_id][feature].median()
        median = 0 if np.isnan(median) else median
        site.update({feature : median})
      
    impute_vals.update({site_id : site})


# In[9]:


impute_vals


# In[10]:


# filling null values
for site_id in range(0, 15+1):
    df = X[X["site_id"]==site_id].fillna(impute_vals[site_id])
    X[X["site_id"]==site_id] = df


# In[11]:


# checking for null vals after imputation
X.isnull().sum()


# In[15]:


# filling null vals in test data from train data impute vals
with open("../data/test_featurized.pkl", "rb") as f:
    test = pickle.load(f)
    
for site_id in range(0, 15+1):
    df = test[test["site_id"]==site_id].fillna(impute_vals[site_id])
    test[test["site_id"]==site_id] = df

del df
gc.collect()


# ### Linear model

# In[24]:


param_grid = {
    'alpha' :[0.0001, 0.01, 0.1]
}

lin = SGDRegressor()
grid = GridSearchCV(lin, param_grid, scoring="neg_mean_squared_error", cv=3)
grid.fit(X ,y)


# In[25]:


grid.best_params_


# In[26]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

linear = SGDRegressor(alpha = 0.01, eta0=1e-5, learning_rate="constant")
linear.fit(scaler.fit_transform(X) ,y)

del X
del train
del y
gc.collect()

save_submission(linear, "_lin", scaler.transform)
save_model(linear,"linear.pkl")


# <img src="../images/model/kaggle/lin.png">

# ### Decision tree

# In[29]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(X, y)

del X
del train
del y
gc.collect()

save_submission(dtr, "_dtr")
save_model(dtr,"dtr.pkl")


# <img src="../images/model/kaggle/dtr.png">

# ### Ada boost

# In[30]:


from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(n_estimators = 1000)
ada.fit(X, y)


# In[31]:


del X
del train
del y
gc.collect()
save_model(ada,"ada.pkl")


# In[36]:


save_submission(ada, "_ada")


# <img src="../images/model/kaggle/ada.png">

# ### MLP

# In[12]:


import tensorflow as tf
import datetime

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.02, random_state=42)

model = tf.keras.Sequential([
    Dense(23, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='relu'),
    Dense(2, activation='relu'),
    Dense(1)
    ])

model.compile(optimizer='adam',
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=tf.keras.losses.MeanSquaredError())

model.fit(X_train, y_train)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X, y, epochs=4, callbacks=[tensorboard_callback],batch_size = 512, validation_data=(X_val, y_val), validation_batch_size=64)


# In[13]:


model.save('mlp')


# In[15]:


save_submission(model, "_mlp")


# <img src="../images/model/kaggle/mlp.png">

# ### Stacking

# In[12]:


with open("ada.pkl", "rb") as f:
    ada = pickle.load(f)

with open("lgbm.pkl", "rb") as f:
    lgbm = pickle.load(f)

with open("linear.pkl", "rb") as f:
    linear = pickle.load(f)
    
def return_train_prediction(model, transform=None):
    
    if transform is None:
        return model.predict(X)
    else:
        return model.predict(transform(X))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
        
base = [ada, lgbm, linear]
transforms=[None, None, scaler.transform]
meta = SGDRegressor(alpha = 0.01, eta0=1e-5, learning_rate="constant")

base_pred = np.zeros((X.shape[0],len(base)), dtype='float')

for i,(m,t) in enumerate(zip(base, transforms)):
    base_pred[:,i] = return_train_prediction(m, t)

meta.fit(base_pred, y)


# In[13]:


save_model(meta,"meta.pkl")


# In[16]:


base_pred = np.zeros((test.shape[0],len(base)), dtype='float')
    
def return_test_prediction(model, transform=None):
    
    if transform is None:
        return model.predict(test.drop(columns=["timestamp","row_id"]))
    else:
        return model.predict(transform(test.drop(columns=["timestamp","row_id"])))
    
for i,(m,t) in enumerate(zip(base, transforms)):
    base_pred[:,i] = return_test_prediction(m, t)

filename = time.strftime("%y-%m-%d-%H%M%S") + "_meta" +".csv"
pd.DataFrame(np.expm1(meta.predict(base_pred)), columns=["meter_reading"]).rename_axis("row_id").to_csv(filename)
print(f"{filename} saved!")
get_ipython().system('p7zip $filename >> /dev/null')


# <img src="../images/model/kaggle/meta.png">
