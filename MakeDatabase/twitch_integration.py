import requests, json
from twitch_api_connection import client_id, refresh_token, token

BASE_URL = 'https://api.twitch.tv/helix/'

HEADERS = {"Client-ID": client_id, 
            'Authorization': f'Bearer {token}',
            "Accept": "application/vnd.v5+json"}
INDENT = 2

# get response from twitch API call
def get_response(query):
  url  = BASE_URL + query
  response = requests.get(url, headers=HEADERS)
  return response

# used for debugging the result
def print_response(response):
  response_json = response.json()
  print_response = json.dumps(response_json, indent=INDENT)
  print(print_response)

# get the current live stream info, given a username
def get_user_streams_query(user_login):
  streams = f'streams?user_login={user_login}'
  return streams

def get_user_query(user_login):
  return f'users?login={user_login}'

def get_user_videos_query(user_id):
  return f'videos?user_id={user_id}&first=50'

def get_games_query():
  return 'games/top'
