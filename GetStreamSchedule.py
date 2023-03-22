#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import pymysql
from datetime import datetime
from datetime import timedelta
import plotly.express as px
from RDS_connection import db, user, password
from mod_streamer_data import get_stream_schedule


# In[2]:


db = pymysql.connect(db, user, password)
cursor = db.cursor()

sql = '''use Streams'''
cursor.execute(sql)

query = '''
SELECT * FROM Streams
'''
#cursor.execute(query)

df = pd.read_sql(query, db, index_col='ID')

#always run this after use
db.commit()
cursor.close()
db.close()


# In[6]:


#example
get_stream_schedule(df, 'scoped','2020-11-05','2020-11-12', plot=True, chunk_after_midnight=True)


# In[ ]:




