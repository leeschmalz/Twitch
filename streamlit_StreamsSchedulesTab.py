import pandas as pd 
import pymysql
from datetime import datetime
from datetime import timedelta
import plotly.express as px
from RDS_connection import db, user, password
from dateutil import tz
from tqdm import tqdm
from mod_streamer_data import get_stream_schedule
from mod_streamer_data import chunk_streams
import streamlit as st
import random

@st.cache
def query_database(query, db, user, password):
    db = pymysql.connect(db, user, password)
    cursor = db.cursor()

    sql = '''use Streams'''
    cursor.execute(sql)

    #cursor.execute(query)

    df = pd.read_sql(query, db, index_col='ID')
    
    #always run this after use
    db.commit()
    cursor.close()
    db.close()

    return df

df = query_database('SELECT * FROM Streams', db, user, password)

#default = random.choice(list(set(list(df['Name']))))

streamer_name = st.text_input("Streamer Name - input streamer name exactly as shown in url twitch.tv/streamer name", '')
start_date = st.text_input("Start Date",str(datetime.now()-timedelta(days=7))[:10])
end_date = st.text_input("End Date",str(datetime.now())[:10])

if (datetime.strptime(end_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(start_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')).days > 7:
    st.write('Time window must be 7 days or less')

if streamer_name != '' and (datetime.strptime(end_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(start_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')).days <= 7:
    stream_schedule_figure = get_stream_schedule(df, streamer_name, start_date, end_date, plot=True, chunk_after_midnight=True)
    st.write(stream_schedule_figure)