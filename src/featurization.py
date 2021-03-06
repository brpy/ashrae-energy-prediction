#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle5 as pickle


# In[2]:


with open("../data/building_df_reduced.pkl","rb") as f:
    building_df = pickle.load(f)

building_df


# In[3]:


building_df.drop(columns="floor_count", inplace=True)
building_df


# In[4]:


with open("../data/train_weather_reduced.pkl", "rb") as f:
    train_weather = pickle.load(f)
    
train_weather


# ### Timestamp alignment

# In[5]:


import datetime

max_temp_hrs = []
for site in range(0,15+1):
    day = pd.Timestamp(datetime.datetime(2016, 1, 1))
    max_temp_hrs_per_day = []
    
    for _ in range(366):
        day_after = day + pd.Timedelta("1 day")
        df = train_weather[(train_weather["site_id"]==site) & (train_weather["timestamp"] >= day) & (train_weather["timestamp"] < day_after)][["timestamp", "air_temperature"]]
        loc = df["air_temperature"].idxmax()
        hour = df.loc[loc]["timestamp"].hour 
        day = day + pd.Timedelta("1 day")
        max_temp_hrs_per_day.append(hour)
    
    max_temp_hrs.append(max_temp_hrs_per_day)


# In[6]:


from scipy.stats import mode

# We align so that all the mode of each sites peak temp time
# has to occur during the afternoon 13:00 to 15:00 time


max_temp = mode(np.array(max_temp_hrs),axis=1).mode.flatten()
sites = ["site_{}".format(i) for i in range(0,16)]
alignment = pd.DataFrame(zip(sites, max_temp),columns=["site_id","peak_temp_hr"])

def correction(hr):
    # peak afternoon hrs are not corrected
    if hr==13 or hr==14 or hr==15:
        return 0
    
    # These sites are in mountain locs and we set peak at 16:00 by -8
    elif hr==0:
        return -8
    
    # For rest of the sites, we align to 15:00
    else:
        return 15-hr

alignment["correction"] = alignment["peak_temp_hr"].apply(correction)
alignment


# In[7]:


# Algning timezone for test data
train_weather_copy = train_weather.copy()
for site in range(0, 16):
    df = train_weather[train_weather["site_id"]==site]["timestamp"].apply(lambda x:x+pd.Timedelta("{} hr".format(alignment.iloc[site]["correction"])))
    train_weather_copy.loc[train_weather_copy["site_id"]==site, "timestamp"] = df

train_weather = train_weather_copy
del train_weather_copy


# In[8]:


with open("../data/train_reduced.pkl", "rb") as f:
    train_reduced = pickle.load(f)

train_reduced


# In[9]:


building_df[building_df["site_id"]==0]


# ### Dropping few site0 data

# In[10]:


# Almost all of readings from this site are zeros in the beggining of the year
# We pick an arbitary date so that most no. of total readings are zeros and drop them from
# the train data.

day = pd.Timestamp(datetime.datetime(2016, 5, 25))
df = train_reduced[(train_reduced["building_id"] < 105) & (train_reduced["timestamp"]<day)]
print("Total no. readings {}".format(len(df)))
print("Total no. of zero reading meters {}".format((df["meter_reading"]==0).sum()))


# In[11]:


train_reduced.drop(df.index, inplace=True)


# ### Hour cyclical + weekend feature

# In[12]:


train_reduced["DT_h_sin"] = train_reduced["DT_hour"].apply(lambda x:np.sin(2*np.pi*x/24))
train_reduced["DT_h_cos"] = train_reduced["DT_hour"].apply(lambda x:np.cos(2*np.pi*x/24))

# Week days are coded as Mon:0,Tue:1,...Fri:4,Sat:5,Sun:6
train_reduced["weekend"] = train_reduced["DT_day_week"].apply(lambda x: int(x>=5))

train_reduced


# In[13]:


train_featurized = train_reduced.merge(building_df, on="building_id", how="left").merge(
    train_weather, on=["site_id","timestamp"], how="left")
train_featurized


# In[14]:


with open("../data/train_featurized.pkl", "wb") as f:
    pickle.dump(train_featurized, f)


# In[15]:


with open("../data/test_reduced.pkl", "rb") as f:
    test_reduced = pickle.load(f)

test_reduced


# In[16]:


test_reduced["DT_h_sin"] = test_reduced["DT_hour"].apply(lambda x:np.sin(2*np.pi*x/24))
test_reduced["DT_h_cos"] = test_reduced["DT_hour"].apply(lambda x:np.cos(2*np.pi*x/24))
test_reduced["weekend"] = test_reduced["DT_day_week"].apply(lambda x: int(x>=5))

test_reduced


# In[17]:


with open("../data/test_weather_reduced.pkl", "rb") as f:
    test_weather = pickle.load(f)


# In[18]:


# timestamp alignment of test data
test_weather_copy = test_weather.copy()
for site in range(0, 16):
    df = test_weather[test_weather["site_id"]==site]["timestamp"].apply(lambda x:x+pd.Timedelta("{} hr".format(alignment.iloc[site]["correction"])))
    test_weather_copy.loc[test_weather_copy["site_id"]==site, "timestamp"] = df

test_weather = test_weather_copy
del test_weather_copy


# In[19]:


test_featurized = test_reduced.merge(building_df, on="building_id", how="left").merge(
    test_weather, on=["site_id","timestamp"], how="left")
test_featurized


# In[20]:


with open("../data/test_featurized.pkl", "wb") as f:
    pickle.dump(test_featurized, f)

