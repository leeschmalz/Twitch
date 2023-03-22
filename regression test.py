#!/usr/bin/env python
# coding: utf-8

# In[1]:


from viewership_prediction import random_forest_regression
import pymysql
from RDS_connection import db, user, password
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta


# In[5]:


#game_id = 33214 #fortnite
game_id = 512710 #warzone
target_streamer = 'swagg'
model, avg_viewership, max_viewership, min_viewership, shared_viewership, rmse, hour_map, day_of_week_map2, X = random_forest_regression(target_streamer, game_id)


# In[2]:


def predict_viewership(target_streamer,game_id):
    model, avg_viewership, max_viewership, min_viewership, shared_viewership, rmse, hour_map, day_of_week_map2, X = random_forest_regression(target_streamer, game_id)

    db1 = pymysql.connect(db, user, password)
    cursor = db1.cursor()

    sql = '''use Streams'''
    cursor.execute(sql)

    #query last 300 rows, then filter within last 10 minutes
    query = f'''
    SELECT * FROM (
        SELECT * FROM Streams WHERE Streams.CurrentGameID = {game_id}
        ORDER BY ID DESC LIMIT 300
    ) sub
    ORDER BY id ASC
    '''

    df_live = pd.read_sql(query, db1, index_col='ID')

    db1.commit()
    cursor.close()
    db1.close()

    conv = []
    for i in range(len(df_live)):
        conv.append(datetime.strptime(df_live['TimestampUTC'].iloc[i], '%Y-%m-%d %H:%M:%S'))
    df_live['TimestampUTC_conv'] = conv

    #if live within last 10 minutes then include
    df_live = df_live[df_live['TimestampUTC_conv'] > datetime.utcnow() - timedelta(minutes = 10)]

    #map current datetime.now().weekday() to string
    day_of_week_map1 = {0 : 'Monday', 1 : 'Tuesday', 2 : 'Wednesday', 3 : 'Thursday', 4 : 'Friday', 5 : 'Saturday', 6 : 'Sunday'}

    counted = []
    competitor_viewership_volume = 0
    for i in range(len(df_live)):
        if df_live['Name'].iloc[i] not in counted and df_live['Name'].iloc[i] != target_streamer:
            counted.append(df_live['Name'].iloc[i])
            competitor_viewership_volume += df_live['ViewerCount'].iloc[i]

    input_to_model = {}
    for col in X.columns[:-3]:
        if col in list(df_live['Name']):
            input_to_model[col] = [1]
        else:
            input_to_model[col] = [0]

    #map temporal ordering to mean viewership ordering that the model was trained on
    input_to_model['HourEncoded'] = [hour_map[datetime.now().hour]]
    input_to_model['DayOfWeekEncoded'] = [day_of_week_map2[day_of_week_map1[datetime.now().weekday()]]]
    input_to_model['CompetitorViewershipVolumeExclusive'] = [competitor_viewership_volume]

    live_viewership_prediction = model.predict(pd.DataFrame(input_to_model))[0]

    import plotly.graph_objects as go

    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = live_viewership_prediction,
        mode = "gauge+number+delta",
        title = {'text': f"{target_streamer}\'s Projected Viewership"},
        delta = {'reference': avg_viewership},
        gauge = {'axis': {'range': [min_viewership, max_viewership]},
                 'steps' : [
                     {'range': [live_viewership_prediction-rmse*2, live_viewership_prediction+rmse*2], 'color': "gray"}]
                }))

    return fig.show()


# In[3]:


#game_id = 33214 #fortnite
#game_id = 512710 #warzone
predict_viewership('innocents',33214)


# In[ ]:




