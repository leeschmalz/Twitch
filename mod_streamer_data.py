import pandas as pd 
import pymysql
from datetime import datetime
from datetime import timedelta
import plotly.express as px
from dateutil import tz
from tqdm import tqdm

def remove_launch_stream_viewership(df, minutes_from_stream_start=10, method='drop'):
    '''
    When a streamer first launches their stream, they will have a low viewership count as viewers start to pile in. 
    We should remove these values for things like tracking average viewership or building predictive models as they are not
    indicative of the success of the attriubutes of the particular stream.
    inputs: dataframe, minutes within stream start to nullify
    output: dataframe with viewership counts within n minutes from stream start nullified
    '''
    drops = []
    for i in tqdm(range(len(df))):
    #use timedelta.seconds * 60 to get minute differential
        if (datetime.strptime(df['TimestampUTC'].iloc[i], '%Y-%m-%d %H:%M:%S') - datetime.strptime(df['StarttimeUTC'].iloc[i], '%Y-%m-%d %H:%M:%S')).seconds < minutes_from_stream_start*60:
            #nullify
            if method == 'nullify':
                df['ViewerCount'].iloc[i] = None
            #drop
            if method == 'drop':
                drops.append(df.iloc[i].name)
    if method == 'drop':
        df.drop(drops,inplace=True)
            
    return df

def add_local_time(df, cols_to_be_converted):
    '''
    adds local timezone to dataframe as new column named f'{col}_toLocal' for each of cols_to_be_coverted
    inputs: dataframe with time columns to be converted (columns should be type str form '%Y-%m-%d %H:%M:%S'),
            list of utc time column names to be converted to local time
    output: dataframe with converted columns added
    '''

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

def crop_timeframe(df, crop_column, start_date, end_date):
    '''
    crops dataframe to specific timeframe
    inputs: dataframe with time column to crop on (column should be type str form '%Y-%m-%d %H:%M:%S'), column to crop on, start_date in UTC time ('%Y-%m-%d %H:%M:%S'), end_date in UTC time ('%Y-%m-%d %H:%M:%S')
    output: cropped dataframe
    '''

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

def get_streams(streamer_data, stream_start_col, timestamp_col):
    '''
    input: dataframe of streamer data, current instance timestamp column, column of current instance stream start time
    output: dictionary of streams with keys Day, Start, StartIndex, Finish, FinishIndex, Titles (list of tuples (Title, timestamp))
    '''
    streams = []
    start_index = streamer_data.iloc[0].name
    for i in range(1, len(streamer_data)):
        if streamer_data[stream_start_col].iloc[i-1] != streamer_data[stream_start_col].iloc[i]:
            #stream ended
            start = streamer_data[stream_start_col].iloc[i-1]
            finish = streamer_data[timestamp_col].iloc[i-1]
            finish_index = streamer_data.iloc[i-1].name
            day = streamer_data[stream_start_col].iloc[i-1].strftime("%Y-%m-%d")
            streams.append(dict(Day=day, Start=start, StartIndex=start_index, Finish=finish, FinishIndex=finish_index))
            start_index = streamer_data.iloc[i].name
            
    for i, stream in enumerate(streams):
        avg_viewership = streamer_data.loc[stream['StartIndex']:stream['FinishIndex']]['ViewerCount'].mean()
        max_viewership = streamer_data.loc[stream['StartIndex']:stream['FinishIndex']]['ViewerCount'].max()
        streams[i]['AvgViewership'] = int(avg_viewership)
        streams[i]['MaxViewership'] = max_viewership
        stream_df = streamer_data.loc[stream['StartIndex']:stream['FinishIndex']]
        unique_titles = []
        titles = []
        for j in range(len(stream_df)):
            if stream_df['StreamTitle'].iloc[j] not in unique_titles:
                titles.append((stream_df['StreamTitle'].iloc[j],stream_df[timestamp_col].iloc[j]))
                unique_titles.append(stream_df['StreamTitle'].iloc[j])
        streams[i]['Titles'] = titles

    return streams

def chunk_streams(streams):
    '''
    if a stream goes through midnight, chunks into two seperate streams cut at the turn of the day.
    input: takes in the output dictionary of get_streams()
    output: dictionary of chunked streams
    '''
    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()
    
    streams_temp = []
    for stream in streams:
        start = stream['Start']
        finish = stream['Finish']
        no_chunks = finish.day - start.day + 1
        for chunk in range(no_chunks):
            if finish.day - start.day != 0:
                temp_finish = start.replace(minute=0, hour=0, second=0, microsecond=0)+timedelta(hours=23.9999)
                temp_finish = temp_finish.replace(tzinfo=to_zone)
                streams_temp.append(dict(Day=str(start.replace(minute=0, hour=0, second=0, microsecond=0))[:10], Start=start, Finish=temp_finish))
                start = start.replace(minute=0, hour=0, second=0, microsecond=0)+timedelta(hours=24)
            else:
                streams_temp.append(dict(Day=str(start.replace(minute=0, hour=0, second=0, microsecond=0))[:10], Start=start, Finish=finish))
    return streams_temp

def get_stream_schedule(df, name, start_date, end_date, plot=True, chunk_after_midnight=True):
    
    '''
    Get previous streams for name from start_date to end_date.
    inputs:
        df: dataframe to get schedule from
        name: name of streamer for which schedule is desired, type str should be exactly as in url twitch.tv/{name}
        start_date: start date of desired schedule window, type str, format '%Y-%m-%d %H:%M:%S'
        end_date: end date of desired schedule window, type str, format '%Y-%m-%d %H:%M:%S'
        plot: plot=True returns plot of schedule, plot=False returns dataframe
        chunk_after_midnight: if True will return streams that pass through midnight as 2 seperate streams, one for each day
        
    output: either a plot of stream schedules depending on plot parameter
    '''

    start_date = start_date + ' 00:00:00'
    end_date = end_date + ' 00:00:00'

    if plot == True and chunk_after_midnight == False:
        raise ValueError('chunk_after_midnight must be True to build desired plotly output.')

    streamer_data = df[df['Name'] == name]

    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()
    current_times = []
    start_times = []
    
    #convert to local time
    for i in range(len(streamer_data)):
        utc = datetime.strptime(streamer_data['TimestampUTC'].iloc[i], '%Y-%m-%d %H:%M:%S')
        utc = utc.replace(tzinfo=from_zone)
        central = utc.astimezone(to_zone)
        current_times.append(central)

        utc = datetime.strptime(streamer_data['StarttimeUTC'].iloc[i], '%Y-%m-%d %H:%M:%S')
        utc = utc.replace(tzinfo=from_zone)
        central = utc.astimezone(to_zone)
        start_times.append(central)

    streamer_data['TimestampLocal'] = current_times
    streamer_data['StartTimeLocal'] = start_times

    utc =  datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    utc = utc.replace(tzinfo=from_zone)
    central = utc.astimezone(to_zone)
    streamer_data = streamer_data[streamer_data['TimestampLocal'] > central]

    utc =  datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    utc = utc.replace(tzinfo=from_zone)
    central = utc.astimezone(to_zone)
    streamer_data = streamer_data[streamer_data['TimestampLocal'] < central]
    
    streams = []
    for i in range(1, len(streamer_data)):
        if streamer_data['StartTimeLocal'].iloc[i-1] != streamer_data['StartTimeLocal'].iloc[i]:
            #stream ended
            start = streamer_data['StartTimeLocal'].iloc[i-1]
            finish = streamer_data['TimestampLocal'].iloc[i-1]
            day = streamer_data['StartTimeLocal'].iloc[i-1].strftime("%Y-%m-%d")
            streams.append(dict(Day=day, Start=start, Finish=finish))

        if chunk_after_midnight:
            streams = chunk_streams(streams) #if streams go past midnight make two different days

        gantt = pd.DataFrame(streams)
        
    if plot:
        for i in range(len(gantt)):
            gantt['Start'].iloc[i] = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').replace(minute=gantt['Start'].iloc[i].minute, hour=gantt['Start'].iloc[i].hour, second=0, microsecond=0)
            gantt['Finish'].iloc[i] = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').replace(minute=gantt['Finish'].iloc[i].minute, hour=gantt['Finish'].iloc[i].hour, second=0, microsecond=0)

        text = []
        for i in range(len(gantt)):
            strt = str(gantt['Start'].iloc[i].time())[:5]
            fin = str(gantt['Finish'].iloc[i].time())[:5]
            text.append(f'{strt} - {fin}')
        gantt['Time'] = text

        fig = px.timeline(gantt, x_start="Start", x_end="Finish", y="Day",text='Time',
                        title=f'{name}\'s Stream Schedule {start_date[:10]} to {end_date[:10]}',labels={'Day':'Date'},
                        range_x=[datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S'),datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')+timedelta(hours=24)])
        #fig.update_layout(xaxis=dict(tickvals= []))
        
        return fig
    
    else:
        return gantt

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