import pandas as pd 
import pymysql
from datetime import datetime
from datetime import timedelta
import plotly.express as px
from RDS_connection import db, user, password
from mod_streamer_data import add_local_time, crop_timeframe, get_streams
from datetime import datetime
from dateutil import tz
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from mod_streamer_data import remove_launch_stream_viewership
import re
import nltk
from nltk.collocations import *
import string
from nltk.corpus import stopwords

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

def count_freq(word, freq):
    for w in word:
        if w in list(freq.keys()):
            freq[w] += 1
        else:
            freq[w] = 1
    return freq

def add_local_time(streamer_data, cols):
    '''
    input: dataframe of streamer data, list of columns to be converted in utc time
    function: adds local timezone to dataframe as new column named f'{col}_toLocal'
    '''
    from datetime import datetime
    from dateutil import tz

    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    #convert to local time
    for col in cols:
        converted_times = []
        for i in range(len(streamer_data)):
            utc = datetime.strptime(streamer_data[col].iloc[i], '%Y-%m-%d %H:%M:%S')
            utc = utc.replace(tzinfo=from_zone)
            central = utc.astimezone(to_zone)
            converted_times.append(central)
            
        streamer_data[f'{col}_toLocal'] = converted_times
    
    return streamer_data

def get_stream_titles_plot(df, streamer_name, start_date, end_date, top_n_keywords):
    streamer_data = df[df['Name'] == streamer_name]
    
    streamer_data = remove_launch_stream_viewership(streamer_data, minutes_from_stream_start=10, method='drop')
    start_date += ' 00:00:00'
    end_date += ' 00:00:00'

    streamer_data = add_local_time(streamer_data,cols=['TimestampUTC','StarttimeUTC'])
    streamer_data = crop_timeframe(streamer_data, start_date=start_date, end_date=end_date, crop_column='TimestampUTC_toLocal')

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()

    table = str.maketrans(dict.fromkeys(string.punctuation))
    freq = {}

    titles_word_list = []
    n_unique_titles = 0
    for i in range(1,len(streamer_data)):
        if streamer_data['StreamTitle'].iloc[i-1] != streamer_data['StreamTitle'].iloc[i]:
            title1 = re.split('!\w*',streamer_data['StreamTitle'].iloc[i-1])
            title = ''
            for item in title1:
                title = title + ' ' + item
            title = title.translate(table).lower().replace('  ',' ').replace('   ',' ')
            title = word_tokenize(title)
            n_unique_titles += 1
            titles_word_list = titles_word_list + ['blank'] + [word for word in title if not word in stopwords.words() and word != 'w']
            
    # change this to read in your data
    finder = TrigramCollocationFinder.from_words(
        titles_word_list)
    finder1 = BigramCollocationFinder.from_words(
        titles_word_list)
    # only bigrams that appear 3+ times
    finder.apply_freq_filter(int(.03*n_unique_titles))
    finder1.apply_freq_filter(int(.03*n_unique_titles))
    # return the 10 n-grams with the highest PMI

    tris = finder.nbest(trigram_measures.pmi, 1000)
    bis = finder1.nbest(bigram_measures.pmi, 1000)

    bis = [word for word in bis if not word[0] == 'blank' and not word[1] == 'blank']
    tris = [word for word in tris if not word[0] == 'blank' and not word[1] == 'blank' and not word[2] == 'blank']

    bis_filtered = bis
    tris_filtered = tris

    #if a bi is a subset of a tri remove it
    drops = []
    for bi in bis_filtered:
        for tri in tris_filtered:
            if ((bi[0] == tri[0] and bi[1] == tri[1]) or (bi[0] == tri[1] and bi[1] == tri[2])):
                drops.append(bi)
    for drop in list(set(drops)):
        bis_filtered.remove(drop)
        
    drops = []
    for tri in tris_filtered:
        for tri1 in tris_filtered:
            if tri[0] == tri1[1] and tri[1] == tri1[2]:
                drops.append(tri)
    for drop in list(set(drops)):
        tris_filtered.remove(drop)

    titles_list_for_plot = []
    freq = {}
    for i in range(1,len(streamer_data)):
        if streamer_data['StreamTitle'].iloc[i-1] != streamer_data['StreamTitle'].iloc[i]:
            title1 = re.split('!\w*',streamer_data['StreamTitle'].iloc[i-1])
            title = ''
            for item in title1:
                title = title + ' ' + item
            title = title.translate(table).lower().replace('  ',' ').replace('   ',' ')
            title = word_tokenize(title)
            if len(title) > 1:
                title = [word for word in title if (not word in stopwords.words()) and word != 'w']
            
            title_phrase_concat = []
            done = False
            if len(title) == 1:
                title_phrase_concat.append(title[0])
            elif len(title) == 2:
                for bi in bis_filtered:
                    if title[0] == bi[0] and title[1] == bi[1]:
                        title_phrase_concat.append(title[0] + ' ' + title[1])
                        done = True
                        break
                if not done:
                    title_phrase_concat.append(title[0])
                    title_phrase_concat.append(title[1])
            else:
                for j in range(2,len(title)):
                    for bi in bis_filtered:
                        if title[j-2] == bi[0] and title[j-1] == bi[1]:
                            title_phrase_concat.append(title[j-2] + ' ' + title[j-1])
                            done = True
                            break
                    
                    for tri in tris_filtered:
                        if title[j-2] == tri[0] and title[j-1] == tri[1] and title[j] == tri[2]:
                            title_phrase_concat.append(title[j-2] + ' ' + title[j-1] + ' ' + title[j])
                            done = True
                            break

                    if not done:
                        title_phrase_concat.append(title[j-2])
                        
                done = False
                for bi in bis_filtered:
                    if title[-2] == bi[0] and title[-1] == bi[1]:
                        title_phrase_concat.append(title[-2] + ' ' + title[-1])
                        done = True
                        break
                        
                if not done:        
                    for word in title_phrase_concat:
                        if title[-2] not in word.split(' '):
                            title_phrase_concat.append(title[-2])
                            break
                    for word in title_phrase_concat:
                        if title[-1] not in word.split(' '):
                            title_phrase_concat.append(title[-1])
                            break

            titles_list_for_plot.append((i-1,title_phrase_concat))
            freq = count_freq(title_phrase_concat,freq=freq)

    sorted_freq = {k: v for k, v in sorted(freq.items(), key=lambda item: item[1], reverse=True)}

    pop = []
    for key in sorted_freq.keys():
        if len(key) < 3:
            pop.append(key)
    for key in pop:
        sorted_freq.pop(key)

    counts_per_word = {}
    for key in list(sorted_freq.keys())[:top_n_keywords]:
        counts_per_word[key] = []
        for tup in titles_list_for_plot:
            title = tup[1]
            title_str = ''
            for word in title:
                title_str = title_str + ' ' + word
            title = title_str
            if (title.find(key) != -1): # if key is substring
                counts_per_word[key].append(streamer_data['ViewerCount'].iloc[tup[0]])
                
    for key in counts_per_word.keys():
        counts_per_word[key] = np.mean(counts_per_word[key])
        
    counts_per_word = {k: v for k, v in sorted(counts_per_word.items(), key=lambda item: item[1], reverse=True)}

    fig = px.bar(x=list(counts_per_word.keys()), y=list(counts_per_word.values()), labels={'x':'Key Word','y':'Average Viewership'},
             title=f'{streamer_name}\'s Average Viewership per Stream Title Keywords')
    fig.add_hline(y=streamer_data['ViewerCount'].mean(),annotation_text="Avg Viewership Overall", 
                annotation_position="top right", line_dash="dot")

    return fig