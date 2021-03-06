#!/usr/bin/env python
# coding: utf-8

# ### ASHRAE Great Energy prediction:
# This competition (https://www.kaggle.com/c/ashrae-energy-prediction/overview) was conducted on Kaggle by ASHRAE (American Society of Heating and Air-Conditioning Engineers) on October 15 2019.
# 
# Modern buildings use a lot of Electrical energy for various applications like heating, cooling, steam and electrical usage. Modern Electrical retrofits can improve the Electrical power efficiency and reduce Electrical consumption and cost. But these retrofits in the scale of a large building can be quite expensive. So, instead of paying the price for retrofits upfront; customers are billed as if the retrofits were not fit for a certain period; till the costs are recovered. Let's assume these variables to be true_cost and assumed_cost ; where true_cost is the actual electrical consumption after retrofits and assumed_cost is the cost that the building would consume if there were no retrofits. The problem at hand is we don't have the assumed_cost information. This could be due to not having enough historical data for the particular building.
# 
# ### Business problem:
# A accurate model would provide better incentives for customers to switch to retrofits. Predicting accurately will smoothen the customer onboarding process and will result in increase in customers. This is because a good model's predictions will result in customers paying the bill amount almost equal to what they would've payed if there were no retrofits; resulting in customers not having to change their expenditures. There are no strict business requirements other than model being highly accurate. Latency requirements are also not high.
# 
# ### Performance metrics:
# 
# This is a standard regression problem so metrics like MSE, R Squared and MAE can be used.
# 
# However, kaggle competition is based on Root Mean Square logarithmic Error.
# 
# Comparison of RMSE and RMSLE,
# 
# Root Mean Square Error (RMSE):
# 

# $$\sqrt{\frac{1}{N}\sum_{i=1}^N{(p_i-a_i)^2}}$$

# Root Mean Square Logarithmic Errror (RMSLE):

# 
# $$\sqrt{\frac{1}{N}\sum_{i=1}^N{(\log(p_i+1)-\log(a_i+1))^2}}$$

# where,

# $p_i$ 
# 
# are predicted values and

# $a_i$ 
# 
# are actual values.

# In[1]:


import numpy as np
import pandas as pd
import os, warnings, math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf','svg')
import scipy
import pickle
warnings.filterwarnings("ignore")


# ## Building Metadata

# In[2]:


building_df = pd.read_csv("../data/building_metadata.csv")
building_df


# In[3]:


sns.countplot(data=building_df,x='site_id')
plt.title("No.of buildings per site")
plt.ylabel("No. of buildings")
plt.show()


# There is a huge variation in no. of buildings wrto. site_id.
# 
# We observe most sites to have about 50-150 buildings.
# 
# Site no. 3 has most no. of buildings.
# 
# Sites 11,7,10,12 have few no. of buildings.

# In[4]:


sns.countplot(data=building_df,x='primary_use')
plt.xticks(rotation=90)
plt.title("No. of buildings by usage")
plt.ylabel("No. of buildings")
plt.show()


# Most buildings are used for Educational purposes and office purposes
# 
# So we might expect majority of buildings to have most electrical load from 9:00 to 18:00
# 
# We might also expect to have most buildings where vacation months have low consumption.

# In[5]:


sns.histplot(building_df,x='square_feet')
plt.title("No. of buildings by area")
plt.ylabel("No. of buildings")
plt.show()


# Most building's area are less than 100,000
# 
# Area of building can correlate with no. of occupants and electrical consumption.

# In[6]:


min_sq_ft = min(building_df['square_feet'])
max_sq_ft = max(building_df['square_feet'])
print("Min Square foot: {}\nMax Square foot: {}".format(min_sq_ft,max_sq_ft))


# In[7]:


sns.histplot(building_df,x='year_built')
plt.title("No. of buildings by year built")
plt.ylabel("No. of buildings")
plt.show()


# We observe a spike in no. of buildings built in around 1960-1980 and after 2000.
# 
# We can expect some correlation with year built and power consumption, since newer equipment is usually more power efficienct.

# In[8]:


sns.histplot(building_df,x='floor_count')
plt.title("No. of buildings by no. of floors")
plt.ylabel("No. of buildings")
plt.show()


# floor count data can give a precise indication of no. of occupants when combined with area of building data.
# 
# But this data has high number of NaN values which prevents us to fully take advantage of this feature. 

# In[9]:


missing_floor_count = building_df["floor_count"].isnull().sum()
missing_floor_percent = missing_floor_count/len(building_df)*100

print("{} buildings ({}%) have floor_count missing".format(missing_floor_count, round(missing_floor_percent,4)))


# ## Weather data

# In[10]:


train_weather_df = pd.read_csv("../data/weather_train.csv")
train_weather_df.head()


# In[11]:


def time_plots_by_site(df,col,save=True,base_dir="../images/eda/",top_dir="",site=None):
    def plot(site):
        fig = px.line(df[df["site_id"]==site], x='timestamp', y=col)
            
        fig.update_layout(
            title="{} at site {}".format(col,site),
            autosize=False,
            width=1800,
            height=600)
        return fig
        
    if save:
        path = os.path.join(base_dir,top_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        
        for _site in df['site_id'].unique():
            fig = plot(_site)
            fig.write_image("{}/{}_site{}.png".format(path,col,_site))
        
    if site is not None:
        fig = plot(site)
        fig.show()


# Saving all plots locally:

# In[12]:


if not os.path.exists("../images/eda/weather"):
    for col in train_weather_df.columns[1:]:
        time_plots_by_site(df=train_weather_df, col=col,top_dir="weather",save=True)


# Air temp. of site 0: (Loaded from disk)

# ![air_temperature_site0.png](attachment:air_temperature_site0.png)

# We observe seasonal effects, where temp. peaks at around June - July / september - october at most sites.
# 
# We also observe daily day-night temperature fluctuations.
# 
# We can expect air temperature to be a strong indicator of power consumption reading since high and low temperature days will result in high usage of cooling and heating systems.

# ## Data Minification:

# In[13]:


# Credit to kyakovlev; https://www.kaggle.com/kyakovlev/ashrae-data-minification
# Minifies dataset so that they use the least amount of memory they could, without data loss

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # Bytes to MB
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                # np.iinfo() returns min and max limit of an int type
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
          
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

##############################################################
def timestamp_to_date(df):#train_df, test_df, train_weather_df, test_weather_df
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df
        
def new_time_features(df):#train_df, test_df
    df['DT_M'] = df['timestamp'].dt.month.astype(np.int8)
    df['DT_W'] = df['timestamp'].dt.weekofyear.astype(np.int8)
    df['DT_D'] = df['timestamp'].dt.dayofyear.astype(np.int16)
    
    df['DT_hour'] = df['timestamp'].dt.hour.astype(np.int8)
    df['DT_day_week'] = df['timestamp'].dt.dayofweek.astype(np.int8)
    df['DT_day_month'] = df['timestamp'].dt.day.astype(np.int8)
    df['DT_week_month'] = df['timestamp'].dt.day/7
    df['DT_week_month'] = df['DT_week_month'].apply(lambda x: math.ceil(x)).astype(np.int8)
    return df
    
def building_transform():
    ########################### Strings to category #########################################
    building_df['primary_use'] = building_df['primary_use'].astype('category')

    ########################### Building Transform ##########################################
    building_df['floor_count'] = building_df['floor_count'].fillna(0).astype(np.int8)
    building_df['year_built'] = building_df['year_built'].fillna(-999).astype(np.int16)

    le = LabelEncoder()
    building_df['primary_use'] = building_df['primary_use'].astype(str)
    building_df['primary_use'] = le.fit_transform(building_df['primary_use']).astype(np.int8)
    
    with open("../models/labelencoder.pkl","wb") as f:
        pickle.dump(le, f)

########################### Base check #################################################
def conversion_and_check(df):#train_df, test_df, building_df, train_weather_df, test_weather_df
    do_not_convert = ['category','datetime64[ns]','object'] #cannot compress further
    original = df.copy()
    df = reduce_mem_usage(df)

    for col in list(df):
        if df[col].dtype.name not in do_not_convert:
            if (df[col]-original[col]).sum()!=0:# Data loss
                df[col] = original[col] #Revert to original
                print('Bad transformation', col)
    return df


# In[14]:


if not os.path.exists("../data/train_reduced.pkl"):
    train_df = pd.read_csv('../data/train.csv')
    train_df = timestamp_to_date(train_df)
    train_df = new_time_features(train_df)
    train_df = conversion_and_check(train_df)


    train_df.to_pickle('../data/train_reduced.pkl')
    print("Sucessfully pickled")
    del train_df


# In[15]:


if not os.path.exists("../data/test_reduced.pkl"):
    test_df = pd.read_csv('../data/test.csv')
    test_df = timestamp_to_date(test_df)
    test_df = new_time_features(test_df)
    test_df = conversion_and_check(test_df)


    test_df.to_pickle('../data/test_reduced.pkl')
    print("Sucessfully pickled")
    del test_df


# In[16]:


if not os.path.exists("../data/building_df_reduced.pkl"):
    building_transform()
    building_df = conversion_and_check(building_df)

    building_df.to_pickle('../data/building_df_reduced.pkl')
    print("Sucessfully pickled")
    del building_df


# In[17]:


if not os.path.exists("../data/train_weather_reduced.pkl"):
    train_weather_df = timestamp_to_date(train_weather_df)
    train_weather_df = conversion_and_check(train_weather_df)

    train_weather_df.to_pickle('../data/train_weather_reduced.pkl')
    print("Sucessfully pickled")
    del train_weather_df


# In[18]:


if not os.path.exists("../data/test_weather_reduced.pkl"):

    test_weather_df = pd.read_csv("../data/weather_test.csv")
    test_weather_df = timestamp_to_date(test_weather_df)
    test_weather_df = conversion_and_check(test_weather_df)


    test_weather_df.to_pickle('../data/test_weather_reduced.pkl')
    print("Sucessfully pickled")
    del test_weather_df


# ## Train data:

# In[19]:


with open("../data/train_reduced.pkl","rb") as pkl:
    train = pd.read_pickle(pkl)
    
train.head()


# All features DT* are added to the original dataset as additional features.
# - DT_M : Month of the year
# - DT_W : Week of the year 
# - DT_D : Day of the year
# - DT_hour : Hour of the day
# - DT_day_week : Day of the week
# - DT_day_month : Day of the month
# - DT_week_month : Week of the month

# In[20]:


meters_per_building = {}
for building in train["building_id"].unique():
    meters_present = np.sort(train[train["building_id"]==building]["meter"].unique())
    all_meters = np.array([False,False,False,False])
    all_meters[meters_present] = True
    meters_per_building.update({building:all_meters})


# In[21]:


meters_per_building_df = pd.DataFrame.from_dict(meters_per_building,orient='index')
meters_per_building_df


# In[22]:


meters_count = pd.DataFrame({"meter_type":meters_per_building_df.sum().index,
"count":meters_per_building_df.sum().values})
meters_count


# No. of buildings with a meter type:

# In[23]:


ax = sns.barplot(x="meter_type",y="count",data=meters_count,hue="meter_type")
ax.set(title="no of meter types per building")
plt.show()


# Buildings with no Meter 0:

# In[24]:


meters_per_building_df[meters_per_building_df[0]==False]


# 1413 buildings have Meter 0, which is the dominant meter.
# 
# Whereas only 498, 324 and 145 buildings have Meter 1, Meter 2 and Meter 3 respectively.
# 
# We can expect that meter 0 to be primary meter and meter 1,2 and 3 to be meters for specialised power usage scenarios.

# In[25]:


def plot_building_meter_reading(building_id):
    fig = px.line(train[train["building_id"]==building_id],x='timestamp', y='meter_reading',color='meter')
    fig.update_layout(
            autosize=False,
            title="Meter Reading at building_{}".format(building_id),
            width=1200,
            height=450)
    return fig


# In[26]:


plot_building_meter_reading(965)


# ![building_965](../images/eda/train/building_965.png)

# ![building_965.png](attachment:building_965.png)

# Meter readings of the building 965 for the entire year. Some of these readings could also be outliers. We can decide on outliers by analysing data of buildings with same primary_use at the same site.

# ### Uni-Variate & Bi-Variate analysis:

# In[27]:


# Mean meter_reading of meter 0 per each building:

avg_meter = train[train["meter"]==0].groupby(by=["building_id"]).mean()["meter_reading"]
avg_meter


# In[28]:


# Merging avg_meter dataframe with building_df

building_meter = pd.DataFrame(building_df.merge(avg_meter,left_on='building_id', right_on='building_id')).set_index("building_id")
building_meter


# Bi-variate analysis of `square_ft` and `meter_reading`

# In[29]:


palette = ["C0", "C1", "C2", "C3","C4", "C5", "C6", "C7", "C8", "C9","C10","C11","C12","C13","C14", "C15"]
x = np.log(building_meter["square_feet"])
y = np.log(building_meter["meter_reading"])

# Since both sqft and meter reading varibales have large scale, we convert to log scale for plotting

ax = sns.scatterplot(x = x,
                     y = y,
                     hue = building_meter["site_id"],palette=palette)
ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
ax.set(xlabel="ln(square_ft)", ylabel="ln(avg_meter_reading)")
ax.set(title="Square ft. of building vs Avg. meter reading\n(both in natural log scale)")
print("Spearman Rank corr: p={:.3f}".format(scipy.stats.spearmanr(x,y)[0]))


# Since,  meter reading data ranges from a low value to a high value, we use natural log of meter_reading data for plotting.
# 
# We are also just taking the mean meter 0 reading for each building to plot since we have daily readings of the data, it is easy to infer from avg meter reading data.
# 
# We see a strong correlation between sqrt ft. and meter reading variable.
# 
# Of course, with increase in area of building, we see increase in consumption.

# Bi-variate analysis of (`square_feet` * `floor_count`) and `meter_reading`

# In[30]:


# floor_info_buildings is a df derived from building_meter df
# with buildings that have non NaN value for floor_count

floor_info_buildings = building_meter[pd.notnull(building_meter["floor_count"])]

x = np.log(floor_info_buildings["square_feet"]*floor_info_buildings["floor_count"])
y = np.log(floor_info_buildings["meter_reading"])

ax = sns.scatterplot(x=x, y=y)
ax.set(xlabel="ln(square_ft * floor_count)", ylabel="ln(avg_meter_reading)")
ax.set(title="Square ft. x floor count of building vs Avg. meter reading\n(both in natural log scale)")
print("Spearman Rank corr: p={:.3f}".format(scipy.stats.spearmanr(x,y)[0]))


# We plot the same plot as above but now we include floor count information.
# 
# We see a slightly higher correlation when we include floor count with sqrft.

# In[31]:


x = building_meter["primary_use"]
y = np.log(building_meter["meter_reading"])

# Since meter reading varibales have large scale, we convert to log scale for plotting

ax = sns.violinplot(x = x,
                     y = y,
                     hue = x)

ax.legend(loc='center right', bbox_to_anchor=(1.75, 0.5), ncol=1)
ax.set(xlabel="primary use", ylabel="ln(avg_meter_reading)")
ax.set(title="Primary use of building vs Avg. meter reading (in natural log scale)")
plt.xticks(rotation=90)
plt.show()


# We observe religious worship to have lower average when compared to toher categories of buildings.
# 
# Educational type buildings have huge fluctuations in mean monthly power consumption data. This could be due to very low consumption in semester break months. And education buildings differ vastly in size and capacity.
# 
# Food sales and service has lowest fluctuation may be beacuse these are operate almost daily for a fixed amount of time.

# Bi-variate analysis of `year_built` and `meter_reading`

# In[32]:


ax = sns.catplot(x = "year_built",
                 y = "meter_reading",
                 data=pd.DataFrame(
                     pd.concat([
                         building_meter["year_built"],
                         np.log(building_meter["meter_reading"])],
                         axis=1)),
                )

ax.set(xlabel="year built", ylabel="ln(avg_meter_reading)")
ax.set(title="year build vs meter Avg. reading (natural log scale)")
locs, labels = plt.xticks()
#locs,labels
plt.xticks(np.array([0, max(locs)*(40/117), max(locs)*(80/117), max(locs)]), ['1900', '1940', '1980', '2017'])
plt.show()


# We don't find any notable pattern with year built feature that affects the avg meter reading.
# 
# So, old and new buildings might not be so different with respect to power consumption. We might expect to see older buildings consuming more power but this would only be true if there were no upgrades to the power equipment.

# In[33]:


# Extracting mean air_temperature per month per site_id
def mean_weather_feature_per_month(feature):
    df = pd.DataFrame(pd.to_datetime(train_weather_df["timestamp"]).dt.month).rename(columns={"timestamp":"DT_M"})
    df = pd.concat([train_weather_df, df], axis=1).loc[:,["site_id", "DT_M",feature]]
    df = df.groupby(by=["site_id", "DT_M"]).mean()
    return df
mean_weather_feature_per_month("air_temperature")


# In[34]:


df = mean_weather_feature_per_month("air_temperature")
palette = ["C0", "C1", "C2", "C3","C4", "C5", "C6", "C7", "C8", "C9","C10","C11","C12","C13","C14", "C15"]
ax = sns.scatterplot(y = "air_temperature",data = df,x="DT_M",hue="site_id",palette=palette)
ax.set(ylabel="avg_air_temp")
ax.set(xlabel="month of year")
ax.set(title="mean air temp per month per site id")
ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)


# We see mean air temp rising in july - august period due to seasonal effects.

# In[35]:


# merging df with site_id,meter_reading with df with building_id

avg_meter_monthly = train[train["meter"]==0].groupby(by=["building_id","DT_M"]).mean()["meter_reading"]
avg_meter_monthly = pd.DataFrame(avg_meter_monthly).rename({"meter_reading":"avg_meter"})
avg_meter_monthly = avg_meter_monthly.join(pd.DataFrame(building_df.loc[:,["site_id","building_id"]]),on="building_id")
avg_meter_monthly


# In[36]:


# merging air_temp data to above df
def avg_by_building(feature):
    df = mean_weather_feature_per_month(feature)
    avg_by_building = avg_meter_monthly.join(df,on=("site_id","DT_M"))
    avg_by_building.drop(columns=["building_id"],inplace=True)
    return avg_by_building.reset_index()
avg_by_building("air_temperature")


# In[37]:


def plot_by_site(site_id,feature):
    _avg_by_building = avg_by_building(feature)
    data = _avg_by_building[_avg_by_building["site_id"]==site_id]
    data["log_meter_reading"] = np.log2(data["meter_reading"])
    ax = sns.lineplot(x="DT_M", y=feature, data=data,color="blue",label="avg_temperature")
    ax = sns.lineplot(x="DT_M", y="meter_reading", data=data, hue="building_id")
    ax.set(title="Avg power consumption & avg {} against month of year for site no. {}".format(feature,site_id))
    ax.set(xlabel="month_of_year", ylabel="Avg power conmuption & Avg air temp")
    ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.5), ncol=1)
    
plot_by_site(12,"air_temperature")


# Here we donot see a observable trend with power consumption and temperature. This could be due to both in low and high temperature conditions power is consumed for heating and cooling respectively.
# 
# We do not observe any particular trend for any of the weather features.

# In[38]:


plot_by_site(7,"cloud_coverage")


# In[39]:


plot_by_site(7,"dew_temperature")


# In[40]:


plot_by_site(7, "precip_depth_1_hr")


# In[41]:


plot_by_site(7, "sea_level_pressure")


# **We donot find any observable trends for weather features and avg power consumption likely due to a weather features being "weak" features and power consumption is strongly affected by general usage pattern by the occupants.**
