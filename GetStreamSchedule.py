def chunk_streams(streams):
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

def get_stream_schedule(name, start_date, end_date):

    '''
    Name type str should be exactly as in url twitch.tv/{name}
    start_date and end_date type str, format '2020-10-16 00:00:00'
    '''
    
    import pandas as pd 
    from datetime import datetime
    from datetime import timedelta
    import plotly.express as px
    from datetime import datetime
    from dateutil import tz

    streamer_data = df[df['Name'] == name]

    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()
    current_times = []
    start_times = []

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

    streams = chunk_streams(streams) #if streams go past midnight make two different days

    gantt = pd.DataFrame(streams)

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
    fig.update_layout(xaxis=dict(tickvals= []))
    return fig.show()