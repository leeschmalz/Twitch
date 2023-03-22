#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import pymysql
from datetime import datetime
from datetime import timedelta
import plotly.express as px
from RDS_connection import db, user, password
from mod_streamer_data import add_local_time, crop_timeframe, get_streams
from datetime import datetime
from dateutil import tz


# In[2]:


db = pymysql.connect(db, user, password)
cursor = db.cursor()

sql = '''use Streams'''
cursor.execute(sql)

query = '''
SELECT * FROM Streams
'''

df = pd.read_sql(query, db, index_col='ID')

#always run this after use
db.commit()
cursor.close()
db.close()


# In[4]:


def add_local_time(df, cols_to_be_converted):
    '''
    inputs: dataframe with time columns to be converted (columns should be type str form '%Y-%m-%d %H:%M:%S'),
            list of utc time column names to be converted to local time
    function: adds local timezone to dataframe as new column named f'{col}_toLocal'
    '''
    from datetime import datetime
    from dateutil import tz

    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    #convert to local time
    for col in cols_to_be_converted:
        converted_times = []
        for i in range(len(df)):
            utc = datetime.strptime(df[col].iloc[i], '%Y-%m-%d %H:%M:%S')
            utc = utc.replace(tzinfo=from_zone)
            central = utc.astimezone(to_zone)
            converted_times.append(central)
            
        df[f'{col}_toLocal'] = converted_times
    
    return df


# In[5]:


def crop_timeframe(df, crop_column, start_date, end_date):
    '''
    inputs: dataframe with time column to crop on (column should be type str form '%Y-%m-%d %H:%M:%S'),
            column to crop on, start_date in UTC time ('%Y-%m-%d %H:%M:%S'), end_date in UTC time ('%Y-%m-%d %H:%M:%S')
    output: cropped dataframe
    '''
    from datetime import datetime
    from dateutil import tz
    
    #define timezones
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()
    
    #crop earlier than start_date
    utc =  datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    utc = utc.replace(tzinfo=from_zone)
    local = utc.astimezone(to_zone)
    df = df[df[crop_column] > local]
    
    #crop later than end_date
    utc =  datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    utc = utc.replace(tzinfo=from_zone)
    local = utc.astimezone(to_zone)
    df = df[df[crop_column] < local]
    
    return df


# In[9]:


name = 'nickeh30'
start_date = '2020-10-16 00:00:00'
end_date = '2020-11-1 00:00:00'
metric = 'avg' #choose from 'avg' or 'max'

streamer_data = df[df['Name'] == name]

streamer_data = add_local_time(streamer_data,cols_to_be_converted=['TimestampUTC','StarttimeUTC'])

streamer_data = crop_timeframe(streamer_data, 
                               start_date=start_date, 
                               end_date=end_date, crop_column='TimestampUTC_toLocal')

streams = get_streams(streamer_data, 
                      timestamp_col='TimestampUTC_toLocal', 
                      stream_start_col='StarttimeUTC_toLocal')


# In[10]:


streams_sorted = []
avg_viewership = 0
for i, stream in enumerate(streams):
    if stream['AvgViewership'] > avg_viewership:
        avg_viewership = stream['AvgViewership']
        ind = i
        
start = streams[ind]['Start']
end = streams[ind]['Finish']
print(f'{name}\'s Best Stream {start_date[:10]} to {end_date[:10]}: \n')
print(f'Start: {start} \nEnd: {end}\n')
for title in streams[ind]['Titles']:
    print(f'{str(title[1])[10:]}   :   {title[0]} \n')

best_stream = streamer_data.loc[streams[ind]['StartIndex']:streams[ind]['FinishIndex']] 

fig = px.bar(best_stream, x='TimestampUTC_toLocal', y='ViewerCount',color='StreamTitle')
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="left",
    x=1
))
fig.update_layout(
    autosize=False,
    width=800,
    height=575,)
fig.show()


# In[ ]:




