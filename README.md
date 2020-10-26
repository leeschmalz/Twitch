# Twitch
## Introduction
The goal of this project is to use scraped twitch data to provide streamers with analytics information that is not already included in their Twitch Analytics dashboard. This will focus largely on information about a streamers competitors allowing them to optimize what, how, and when they stream to maximize viewership.
## Data Description
### Gathering
The current state of the dataset includes the top 300 streamers for each of the top 5 most popular games streamed on Twitch overall resulting in a dataset that includes 1500 streamers, this can be easily expanded in the future. The data stream feeds the database every 5 minutes and includes a data point for each of the 1500 streamers that are currently live. The list of streamers in question comes from a webscrape of twitchmetrics.net and the data itself comes from the Twitch API. These processes are running on an AWS EC2 t2.micro instance that feeds an AWS RDS database.
## The dataset
After one month of data gathering, we have a 258 MB dataframe in RDS with approximately ~2,000,000 instances of data. Each instance contains a unique instance ID as a primary key, along with the streamer's name, the current timestamp (UTC), the current game ID, the stream title, viewer count, and stream start timestamp (UTC).
