#!/usr/bin/env python
# coding: utf-8

### Matt Newkirk ~ UW FinTech ~ Project 2: Doge Pirates of the Cryptobean
### This script reads the Kraken API for BTC/USD data approx every 5 min and appends average Twitter & Reddit Sentiment.

#Import libs

import requests
import sys
import pandas as pd
import os
import numpy as np
from datetime import datetime
import ccxt
import time
from dotenv import load_dotenv
import tweepy #A wrapper for Twitter API.
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import asyncio
import statistics #For taking the average of a list.

def spinning_cursor(): ###https://stackoverflow.com/questions/4995733/how-to-create-a-spinning-command-line-cursor
    while True:
        for cursor in '|/-\\':
            yield cursor

async def tweet_search_sent(string_query, number_tweets = 500):
    load_dotenv()
    consumer_key = os.getenv('TWITTER_API_KEY')
    consumer_secret = os.getenv('TWITTER_SECRET_KEY')
    auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
    api = tweepy.API(auth)
    
    print("Analyzing Tweets")
    sentiment = []
    df = pd.DataFrame()
    spinner = spinning_cursor()
    for tweets in tweepy.Cursor(api.search, q=string_query).items(number_tweets):
        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')
        contents = tweets.text
        sentiment.append(contents)
        #sentiment.append(sentiment_reader_comp(contents))
    df = pd.DataFrame(sentiment, columns=['text'])
    df['text'] = df['text'].apply(sentiment_reader_comp)
    return df['text'].mean()




def sentiment_reader_comp(text_block):
    analyzer = SentimentIntensityAnalyzer()
    text_sentiment = analyzer.polarity_scores(text_block)
    sentiment =  text_sentiment["compound"]
    return sentiment




async def get_redd_sent(subreddit):
    url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&sort=desc&sort_type=created_utc&size=1000"
    request = requests.get(url)
    json_response = request.json()
    df_response = pd.json_normalize(json_response['data'])
    df_response = df_response[df_response['selftext'] != '[removed]']
    df_response = df_response[df_response['selftext'] != '']
    df_response['sent'] = df_response['selftext'].apply(sentiment_reader_comp)
    return df_response['sent'].mean()


# In[9]:


async def get_kraken(ticker = "BTC/USD"):
    data = {}
    load_dotenv()
    kraken_public_key = os.getenv("KRAKEN_PUBLIC_KEY")
    kraken_secret_key = os.getenv("KRAKEN_SECRET_KEY")
    kraken = ccxt.kraken({"apiKey": kraken_public_key, "secret": kraken_secret_key})
    data = kraken.fetch_ticker(ticker)
    return data




async def main(loop):
    x = 0
    
    global df
    global dict_btc
    df = pd.DataFrame()
    loops = 500
    avg_time = 0
    while x <= loops:
        wait = 300 #seconds
        start = time.time()
        try:
            print('Collecting Ticker Data')
            dict_btc = await get_kraken("BTC/USD")
            print(f"Current close for BTC/USD: ${dict_btc['close']}")
        except:
            dict_btc = {}
            print(f"Kraken fail at {x}")
        #await asyncio.sleep(2)
        
        try:
            print("Gathering Tweets")
            twi_sent = await tweet_search_sent('bitcoin OR btc')
            print(f"Average compound Vader score of last 500 tweets about bitcoin {twi_sent}")
        except:
            twi_sent = 0
            print(f"Tweet fail at {x}")
        #await asyncio.sleep(2)
        try:
            print("Reading Reddit")
            redd_sent = await get_redd_sent('bitcoin')
            print(f"Average compound Vader score of roughly last 100 reddit posts in r/bitcoin {redd_sent}")
        except:
            redd_sent = 0
            print(f"Reddit fail at {x}")
        #await asyncio.sleep(2)
        try:
            del dict_btc['info']
        except:
            print("Empty Dict")
        
        dict_btc['twi_sent'] = twi_sent
        dict_btc['redd_sent'] = redd_sent
        df = df.append(dict_btc, ignore_index=True)
        x += 1
        
        print(f"DataFrame Shape: {df.shape}")
        end = time.time()
        runtime = end - start ###https://www.blog.pythonlibrary.org/2016/05/24/python-101-an-intro-to-benchmarking-your-code/
        msg = "The runtime for this loop took {time} seconds to complete"
        print(msg.format(time=runtime))
        avg_time += runtime
        wait -= int(round(runtime,0))
        print("Remaining Wait: (ctrl+c to break) ")
        for i in range(wait): ###https://stackoverflow.com/questions/25189554/countdown-clock-0105/50148334
            mins, secs = divmod(wait-i, 60)
            timeformat = '{:02d}:{:02d}'.format(mins, secs)
            #timeformat = ':{:02d}'.format(wait-i)
            print(timeformat, end='\r')
            await asyncio.sleep(1)
    avg_time = avg_time / x
    msg = "The average runtime for each loop was {time} seconds to complete"
    print(msg.format(time=avg_time))

loop = ""
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(main(loop))
except KeyboardInterrupt:
    df.to_csv('five_min_intervals_example.csv')
except:
    df.to_csv('five_min_intervals_example_error.csv')
df.to_csv('five_min_intervals_example_x.csv')
