from RDS_connection import db, user, password
import matplotlib.pyplot as plt
import pymysql
import pandas as pd 
import numpy as np
from mod_streamer_data import add_local_time, remove_launch_stream_viewership
from tqdm import tqdm
from datetime import datetime
from dateutil import tz
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
pd.options.mode.chained_assignment = None 

def plot_feature_importances(model, X):
    n_features = 20
    feature_imp = dict(zip(list(X.columns),model.feature_importances_))
    feature_imp = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: item[1],reverse=True)}
    feature_imp = dict(list(feature_imp.items())[:20])
    
    plt.barh(np.arange(n_features), list(feature_imp.values()), align='center')
    plt.yticks(np.arange(n_features),list(feature_imp.keys()))
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)

def random_forest_regression(target_streamer, game_id):
    db1 = pymysql.connect(db, user, password)
    cursor = db1.cursor()

    sql = '''use Streams'''
    cursor.execute(sql)

    query = f'''
    SELECT *

    FROM Streams
    WHERE Streams.CurrentGameID = {game_id} 
    '''

    df_competitors = pd.read_sql(query, db1, index_col='ID')

    #always run this after use
    db1.commit()
    cursor.close()
    db1.close()
    
    shared_viewership = list(df_competitors['Name'].unique())
    #shared_viewership.remove(target_streamer)
    df_competitors = remove_launch_stream_viewership(df_competitors, minutes_from_stream_start=30, method='drop')
    target_streamer_data = df_competitors[df_competitors['Name'] == target_streamer]
    target_streamer_data = add_local_time(target_streamer_data, cols_to_be_converted=['TimestampUTC','StarttimeUTC'])

    #create dummy columns for each competitor to be overwritten
    for name in shared_viewership:
        target_streamer_data[name] = [0]*len(target_streamer_data)

    #because I built the database with string dates... oops
    conv = []
    for i in tqdm(range(len(df_competitors))):
        conv.append(datetime.strptime(df_competitors['TimestampUTC'].iloc[i], '%Y-%m-%d %H:%M:%S'))
    df_competitors['TimestampUTC_conv'] = conv


    for name in shared_viewership:
        target_streamer_data[name] = [0]*len(target_streamer_data)
        
    total_competitor_viewership_inc = []
    total_competitor_viewership_exc = []
    for i in range(len(target_streamer_data)):
        window_start = datetime.strptime(target_streamer_data['TimestampUTC'].iloc[i], '%Y-%m-%d %H:%M:%S') - timedelta(minutes=3)
        window_end = datetime.strptime(target_streamer_data['TimestampUTC'].iloc[i], '%Y-%m-%d %H:%M:%S') + timedelta(minutes=3)
        window_df = df_competitors[df_competitors['TimestampUTC_conv'] > window_start]
        window_df = window_df[window_df['TimestampUTC_conv'] < window_end]

        #make sure all names are unique, since database loads once per 5 minutes, this gets one instance from every streamer in the
        #database that is currently live streaming the game in question
        names = []
        drops = []
        for j in range(len(window_df)):
            if window_df['Name'].iloc[j] not in names:
                names.append(window_df['Name'].iloc[j])
            else:
                drops.append(window_df.index[j])
                
        window_df.drop(index=drops, inplace=True)
        
        for name in window_df['Name']:
            target_streamer_data[name].iloc[i] = 1 
        
        #including target streamer
        total_competitor_viewership_inc.append(window_df['ViewerCount'].sum() + target_streamer_data['ViewerCount'].iloc[i])
        #excluding target streamer
        total_competitor_viewership_exc.append(window_df['ViewerCount'].sum())
        
    target_streamer_data[f'CompetitorViewershipVolumeInclusive'] = total_competitor_viewership_inc
    target_streamer_data[f'CompetitorViewershipVolumeExclusive'] = total_competitor_viewership_exc
    
    mean = target_streamer_data['ViewerCount'].mean()
    std = target_streamer_data['ViewerCount'].std()
    target_streamer_data = target_streamer_data[target_streamer_data['ViewerCount'] < mean + std*2.5]

    day_of_week_map = {0 : 'Monday', 1 : 'Tuesday', 2 : 'Wednesday', 3 : 'Thursday', 4 : 'Friday', 5 : 'Saturday', 6 : 'Sunday'}

    hour = []
    day_of_week = []
    for i in range(len(target_streamer_data)): 
        day_of_week.append(day_of_week_map[target_streamer_data['TimestampUTC_toLocal'].iloc[i].weekday()])
        hour.append(target_streamer_data['TimestampUTC_toLocal'].iloc[i].hour)

    target_streamer_data['Hour'] = hour
    target_streamer_data['DayOfWeek'] = day_of_week

    #random forest needs categorical variable encoded in order by viewership count. DayOfWeek --> DayOfWeekEncoded, Hour --> HourEncoded
    hour_map = {}
    for hour in list(target_streamer_data['Hour'].unique()):
        hour_map[hour] = target_streamer_data[target_streamer_data['Hour'] == hour]['ViewerCount'].mean()
        hour_map = {k: v for k, v in sorted(hour_map.items(), key=lambda item: item[1])}
    for i, hour in enumerate(hour_map.keys()):
        hour_map[hour] = i
    target_streamer_data['HourEncoded']= target_streamer_data['Hour'].map(hour_map)

    dayofweek_map = {}
    for day in list(target_streamer_data['DayOfWeek'].unique()):
        dayofweek_map[day] = target_streamer_data[target_streamer_data['DayOfWeek'] == day]['ViewerCount'].mean()
        dayofweek_map = {k: v for k, v in sorted(dayofweek_map.items(), key=lambda item: item[1])}
    for i, day in enumerate(dayofweek_map.keys()):
        dayofweek_map[day] = i
    target_streamer_data['DayOfWeekEncoded'] = target_streamer_data['DayOfWeek'].map(dayofweek_map)

    X = target_streamer_data[shared_viewership + ['HourEncoded','DayOfWeekEncoded','CompetitorViewershipVolumeExclusive']]
    y = target_streamer_data['ViewerCount']


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42, shuffle=False)

    from sklearn.metrics import classification_report
    rfreg = RandomForestRegressor(n_estimators=1000, n_jobs=-1) #max_depth=depth, n_estimators=est)
    rfreg.fit(X_train, y_train)
    preds = rfreg.predict(X_test)
    #acc.append(accuracy_score(preds, y_test))

    print('RMSE')
    print(np.sqrt(mean_squared_error(preds, y_test)))

    sns.scatterplot(x=preds,y=y_test)
    plt.ylabel('Actual Viewer Count')
    plt.xlabel('Predicted Viewer Count')
    plt.title(f'{target_streamer}\'s predicted vs actual viewership: \n')
    plt.show()

    plot_feature_importances(rfreg, X)
    plt.title('Most Important Features')
    plt.show()

    return rfreg, target_streamer_data['ViewerCount'].mean(), target_streamer_data['ViewerCount'].max(), target_streamer_data['ViewerCount'].min(), shared_viewership, np.sqrt(mean_squared_error(preds, y_test)), hour_map, dayofweek_map, X
