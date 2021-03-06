import flask
from flask import Flask, jsonify, request
from waitress import serve
import pandas as pd
import pickle
import numpy as np
import math
import lightgbm

app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template("index.html")

def timefeatures(timestamp):
    # time features

    DT_M = timestamp.dt.month.astype(np.int8)
    DT_W = timestamp.dt.isocalendar().week.astype(np.int8)
    DT_D = timestamp.dt.dayofyear.astype(np.int16)
    
    DT_hour = timestamp.dt.hour.astype(np.int8)
    DT_day_week = timestamp.dt.dayofweek.astype(np.int8)
    DT_day_month = timestamp.dt.day.astype(np.int8)
    DT_week_month = timestamp.dt.day/7
    DT_week_month = DT_week_month.apply(lambda x: math.ceil(x)).astype(np.int8)
    
    DT_h_sin = DT_hour.apply(lambda x:np.sin(2*np.pi*x/24))
    DT_h_cos = DT_hour.apply(lambda x:np.cos(2*np.pi*x/24))

    # Week days are coded as Mon:0,Tue:1,...Fri:4,Sat:5,Sun:6
    weekend = DT_day_week.apply(lambda x: int(x>=5))
    
    return list(map(lambda x:x.to_numpy()[0],[DT_M, DT_W, DT_D, DT_hour, DT_day_week, DT_day_month, DT_week_month, DT_h_sin, DT_h_cos, weekend]))

def weatherfeatures(response):
    none_check_float = lambda x: None if not x else float(x)
    air_temperature = response['air_temperature']
    cloud_coverage = response['cloud_coverage']
    dew_temperature = response['dew_temperature']
    precip_depth_1_hr = response['dew_temperature']
    sea_level_pressure = response['sea_level_pressure']
    wind_direction = response['wind_direction']
    wind_speed = response['wind_speed']
    return list(map(none_check_float, [air_temperature,cloud_coverage,dew_temperature,precip_depth_1_hr,sea_level_pressure,wind_direction,wind_speed]))

@app.route("/predict", methods=['POST'])
def predict():
    response = request.form.to_dict()

    with open("./models/labelencoder.pkl", "rb") as f:
        le = pickle.load(f)

    with open("./models/lgbm.pkl", "rb") as f:
        lgbm = pickle.load(f)
    

    building_metadata = pd.read_csv("./data/building_metadata.csv")
    building_id = int(response["building"])
    
    meter = int(response["meter"])
    site_id = int(response["site"])
    
    df = building_metadata[building_metadata["building_id"]==building_id].to_dict(orient="list")
    df = {k:v[0] for k,v in df.items()}

    primary_use = int(le.transform(np.array(df['primary_use']).reshape(1,)).astype(np.int8)[0])
    
    square_feet = df['square_feet']
    square_feet = None if np.isnan(square_feet) else square_feet
    
    year_built = df['year_built']
    year_built = None if np.isnan(year_built) else year_built

    response.update({"primary_use" : primary_use, "square_feet" : square_feet, "year_built" : year_built})
    
    try:
        timestamp = pd.to_datetime(pd.Series(response["timestamp"]))
    except:
        return f'{response["timestamp"]} is Invalid timestamp!'
    
    encoded = [building_id] + [meter] + timefeatures(timestamp) + [site_id, primary_use, square_feet, year_built] + weatherfeatures(response)
    meter_reading = np.expm1(lgbm.predict(np.array(encoded).reshape(1,-1))[0])
    return {"meter_reading" : meter_reading}

if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=8080)
    serve(app, host="0.0.0.0", port=8888)
    

