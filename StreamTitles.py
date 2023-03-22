#!/usr/bin/env python
# coding: utf-8

# In[1]:


from get_stream_titles_plot import *
from RDS_connection import db, user, password
import pymysql


# In[2]:


db = pymysql.connect(db, user, password)
cursor = db.cursor()

sql = '''use Streams'''
cursor.execute(sql)

query = '''
SELECT * FROM Streams
where Streams.Name = 'nickmercs'
'''

df = pd.read_sql(query, db, index_col='ID')

#always run this after use
db.commit()
cursor.close()
db.close()


# In[7]:


fig = get_stream_titles_plot(df, 'nickmercs', '2020-10-15', '2020-11-19', 20)


# In[8]:


fig.show()


# In[ ]:




