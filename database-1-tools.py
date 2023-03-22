#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Current start date: 9/23 2:30PM, scheduled to run 2 months
import pandas as pd 
import pymysql
from datetime import datetime
from datetime import timedelta
import plotly.express as px
from RDS_connection import db, user, password


# # Tools to query from database-1

# In[10]:


db = pymysql.connect(db, user, password)
cursor = db.cursor()

sql = '''use Streams'''
cursor.execute(sql)


# In[11]:


query = '''
SELECT * FROM Streams
'''
#cursor.execute(query)


# In[12]:


df = pd.read_sql(query, db, index_col='ID')


# In[13]:


#always run this after use
db.commit()
cursor.close()
db.close()


# In[14]:


df.head()


# In[7]:


#df.to_csv('RDS_snapshot_10-23-20')


# # Tools to modify database / tables

# In[110]:


db = pymysql.connect('database-1.cfcym0vwtoiw.us-east-1.rds.amazonaws.com', 'admin', 'EulerDickson2140$')
cursor = db.cursor()

sql = '''use Streams'''
cursor.execute(sql)


# In[111]:


sql = "DROP TABLE Streams"
cursor.execute(sql)


# In[112]:


#Create Database
sql = '''create database Streams'''
#cursor.execute(sql)
#cursor.connection.commit()


# In[113]:


#Create Table
create_table_query = '''CREATE TABLE Streams (
    ID int NOT NULL,
    Name varchar(45) COLLATE utf8_bin NOT NULL,
    TimestampUTC varchar(45) NOT NULL,
    CurrentGameID varchar(45) NOT NULL,
    StreamTitle varchar(255) NOT NULL,
    ViewerCount int(10) NOT NULL,
    StarttimeUTC varchar(45),
    PRIMARY KEY (ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin
AUTO_INCREMENT=1;'''

cursor.execute(create_table_query)


# In[114]:


#always run this after use
db.commit()
cursor.close()
db.close()


# In[ ]:




