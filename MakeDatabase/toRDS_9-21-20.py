import twitch_integration
from get_top_streamers import get_top_streamers
import pandas as pd 
from datetime import datetime
from dateutil import tz
import dateutil.parser
import time
import pymysql
import json
from tqdm import tqdm
from RDS_connection import db, user, password

def getDateTime(s):
    d = dateutil.parser.isoparse(s)
    return d

games = ['Fortnite','Among+Us','Call+Of+Duty%3A+Modern+Warfare','League+of+Legends','Fall+Guys%3A+Ultimate+Knockout','Counter-Strike%3A+Global+Offensive']

#if a streamer is in top 150 for multiple games, make sure we only grab data once per cycle
users_not_unique = get_top_streamers(games)
users = []

for user in users_not_unique:
    if user not in users:
        users.append(user)

db = pymysql.connect(db, user, password)
#cursor = db.cursor()

sql = '''use Streams'''
#cursor.execute(sql)

create_table_query = '''CREATE TABLE Streams (
    ID int NOT NULL,
    Name varchar(255) COLLATE utf8_bin NOT NULL,
    TimestampUTC varchar(45) NOT NULL,
    Top100Game varchar(45) NOT NULL,
    CurrentGameID varchar(45) NOT NULL,
    ViewerCount int(10) NOT NULL,
    Live int(10) NOT NULL,
    StarttimeUTC varchar(45),
    PRIMARY KEY (ID)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin
AUTO_INCREMENT=1;'''

#uncomment to create table
#cursor.execute(create_table_query)

count = 0
prim_id = 180387
while True:
    if datetime.now().minute % 5 == 0 and datetime.now().second == 0: #run every 5 mins
        start = datetime.utcnow()
        print('db open')
        db = pymysql.connect(db, user, password)
        cursor = db.cursor()

        sql = '''use Streams'''
        cursor.execute(sql)

        if count > 17251:
            break
        count += 1
        print('gathering...')
        #gets all streamers in top 100 for each game current stream status
        
        for user in tqdm(users):
            now = str(datetime.utcnow()).split('.')[0]
            day = datetime.now().day
            query = twitch_integration.get_user_streams_query(user)
            try:
                data = twitch_integration.get_response(query).json()
                if len(data['data']) != 0: #if live
                    title = data['data'][0]['title'].replace('\'','')
                    current_game = str(data['data'][0]['game_id'])
                    viewer_count = data['data'][0]['viewer_count']
                    start_time = str(getDateTime(data['data'][0]['started_at'])).split('+')[0]
                    prim_id += 1              
                    #insert into db
                    insertStatement = f"INSERT INTO Streams (ID, Name, TimestampUTC, CurrentGameID, StreamTitle, ViewerCount, StarttimeUTC) VALUES ({prim_id},\'{user}\',\'{now}\',\'{current_game}\',\'{title}\',{viewer_count},\'{start_time}\')"   
                    cursor.execute(insertStatement)
            except Exception as e:
                print('Couldnt gather data')
                print(e)
                pass

        end = datetime.utcnow()
        print(f'took {end-start} seconds')
        db.commit()
        cursor.close()
        db.close()
        print('Rows commited and db closed')
        
