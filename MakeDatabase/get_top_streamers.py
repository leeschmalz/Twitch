import bs4 as bs
import requests
import re

def get_top_streamers(games):
    '''
    Returns dictionary of top streamers last 30 days by watched hours per game
    {game : [list of streamers]}

    enter games as web address format 
    example: games = ['Fortnite','Among+Us','Call+Of+Duty%3A+Modern+Warfare'] 'Call+Of+Duty%3A+Modern+Warfare'
    '''
    top_streamers = {}
    #get page 1
    for game in games:
        r = requests.get(f'https://www.twitchmetrics.net/channels/viewership?game={game}')

        soup = str(bs.BeautifulSoup(r.content,features='lxml'))
        top_streamers[game] = re.findall(r'https://www.twitch.tv/(.+?)"&gt', soup)

        for page in range(2,6):
        #get pages 2-5
            r = requests.get(f'https://www.twitchmetrics.net/channels/viewership?game={game}&page={page}')

            soup = str(bs.BeautifulSoup(r.content,features='lxml'))
            top_streamers[game] = top_streamers[game] + re.findall(r'https://www.twitch.tv/(.+?)"&gt', soup)

    top_streamers_list = []
    for game in games:
        top_streamers_list = top_streamers_list + top_streamers[game]

    return top_streamers_list