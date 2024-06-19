#!/usr/bin/env python
# coding: utf-8


import pickle
import os
import sys
import pandas as pd

year = int(sys.argv[1])  # 2023
month = int(sys.argv[2])  # 3
taxi_type = str(sys.argv[3])  # 'yellow'



with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/{taxi_type}_{year:04d}_{month:02d}.parquet'




categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


df = read_data(input_file)



dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)



x = y_pred.mean()
x = str(x)
print('predicted_mean_duration: ' + x)


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
 

df_result = pd.DataFrame()


# Create a DataFrame
df_result['ride_id'] = df['ride_id']
df_result['predicted_durtion'] = y_pred

# Display the DataFrame (optional)
print("DataFrame with Results:")
print(df_result)



df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[25]:


file_size = os.path.getsize(output_file)

print(f"Size of 'results.csv' file: {file_size} bytes")


# In[24]:


len(output_file)


# In[ ]:




