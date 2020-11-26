import pandas as pd 
import pymysql
from datetime import datetime
from datetime import timedelta
import plotly.express as px
from RDS_connection import db, user, password
from dateutil import tz
from tqdm import tqdm
from mod_streamer_data import chunk_streams
import streamlit as st
import random
from get_stream_titles_plot import *

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

streamer_name = st.text_input("Streamer Name - input streamer name exactly as shown in url twitch.tv/streamer name", '')
start_date = st.text_input("Start Date",str(datetime.now()-timedelta(days=31))[:10])
end_date = st.text_input("End Date",str(datetime.now())[:10])
top_n_keywords = st.slider('Number of Keywords',5,30,20,1)

if (datetime.strptime(end_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(start_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')).days < 29:
    st.write('Time window must be at least 1 month for accurate results')

if streamer_name != '' and (datetime.strptime(end_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S') - datetime.strptime(start_date + ' 00:00:00', '%Y-%m-%d %H:%M:%S')).days >= 29:
    fig = get_stream_titles_plot(df, streamer_name, start_date, end_date, top_n_keywords)
    fig.show()
    st.plotly_chart(fig, use_container_width=False)