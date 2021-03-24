#!/usr/bin/env python
# coding: utf-8

### Matt Newkirk ~ UW FinTech ~ Project 2: Doge Pirates of the Cryptobean
### This script reads the Kraken API for BTC/USD data approx every 5 min and appends average Twitter & Reddit Sentiment.

#Import libs

import requests
import sys
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' ###https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import numpy as np
from datetime import datetime
import ccxt
import time
from dotenv import load_dotenv
import tweepy #A wrapper for Twitter API.
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import asyncio
from matplotlib import pyplot
import pyfiglet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import Callback
#import warnings
#warnings.filterwarnings('ignore')

#This function is a simple spinning cursor.
def spinning_cursor(): ###https://stackoverflow.com/questions/4995733/how-to-create-a-spinning-command-line-cursor
    while True:
        for cursor in '|/-\\':
            yield cursor


#This function calls the Twitter API using Tweepy, an UnOfficial API wrapper library.           
#"""Delete the # to disable Twitter Calls
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
        
        ###Spinning Progress Bar
        sys.stdout.write(next(spinner))
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')
        ### /Spinning Progress Bar
        
        contents = tweets.text
        sentiment.append(contents)
    df = pd.DataFrame(sentiment, columns=['text'])
    df['text'] = df['text'].apply(sentiment_reader_comp)
    return df['text'].mean()
#"""


#This function applies VADER sentiment analysis to a block of text and returns the Compound score.
def sentiment_reader_comp(text_block):
    analyzer = SentimentIntensityAnalyzer()
    text_sentiment = analyzer.polarity_scores(text_block)
    sentiment =  text_sentiment["compound"]
    return sentiment



#This function uses an UnOfficial Reddit API to retrieve the most recent 100 posts/comments from r/Bitcoin.
#After filtering out posts/comments without text, the true count is more like 50 posts/comments.
async def get_redd_sent(subreddit):
    url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&sort=desc&sort_type=created_utc&size=1000"
    request = requests.get(url)
    json_response = request.json()
    df_response = pd.json_normalize(json_response['data'])
    df_response = df_response[df_response['selftext'] != '[removed]']
    df_response = df_response[df_response['selftext'] != '']
    df_response['sent'] = df_response['selftext'].apply(sentiment_reader_comp)
    return df_response['sent'].mean()

#This function calls the Kraken API using the CCXT library to fetch the most recent ticker data.
async def get_kraken(ticker = "BTC/USD"):
    data = {}
    load_dotenv()
    kraken_public_key = os.getenv("KRAKEN_PUBLIC_KEY")
    kraken_secret_key = os.getenv("KRAKEN_SECRET_KEY")
    kraken = ccxt.kraken({"apiKey": kraken_public_key, "secret": kraken_secret_key})
    data = kraken.fetch_ticker(ticker)
    return data

#This Function prepares fresh ticker data for processing in the LSTM model.
async def live_ticker_shaper(dataset):
    dataset['datetime'] = pd.to_datetime(dataset['datetime'],infer_datetime_format = True)
    dataset.set_index(['datetime'],inplace = True)
    data = dataset[['close','baseVolume','redd_sent','twi_sent']].copy()
    values = data.values
    values = values.astype('float32')
    return values

#This function initializes our LSTM model that will be used to make inferences.
def generate_model(path , steps, features):
    model = Sequential()
    model.add(LSTM(50, input_shape=(steps, features),return_sequences=True))
    #model.add(Dropout(.2))
    model.add(LSTM(units=50, return_sequences=False))
    #model.add(Dropout(.2))
    #model.add(LSTM(units=50, return_sequences=True))
    #model.add(Dropout(.2))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.load_weights(path)
    return model

#Defines a class for custom model fit progress message.
import tensorflow as tf
class PrintLogs(tf.keras.callbacks.Callback):
    def __init__(self, epochs):
            self.epochs = epochs
    
    def set_params(self, params):
            params['epochs'] = 0
            params['verbosity'] = 0

    def on_epoch_begin(self, epoch, logs=None):
            print('Epoch %d/%d' % (epoch + 1, self.epochs), end='\r')

#Function for updating model with the most recent pull of data.
async def live_fitter(model, train_X, train_y):
    
    try:
        epochs = 1000
        model.fit(train_X, train_y, epochs=epochs, batch_size=1, verbose=0, shuffle=False, callbacks=[PrintLogs(epochs)])
        print('Epochs: ')
    except:
        print('Model Fit Failure')


# convert time-series to supervised learning dataset (Generates input and output lags.)
###https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.fillna(0,inplace=True)
	return agg


async def main(loop):
    #Initialize Important Variables
    x = 0
    global df
    global older
    global dict_btc
    global recent
    global scaler
    df = pd.DataFrame()
    loops = 1000
    avg_time = 0
    dataset = pd.read_csv('data/500_five_min_intervals_btc.csv')
    dataset['datetime'] = pd.to_datetime(dataset['datetime'],infer_datetime_format = True)
    dataset.set_index(['datetime'],inplace = True)
    data = dataset[['close','baseVolume','redd_sent','twi_sent']].copy()
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    dataset['close'] = y_scaler.fit_transform(dataset['close'].values.reshape(-1,1))
    vol_scaler = MinMaxScaler(feature_range=(0, 1))
    dataset['baseVolume'] = vol_scaler.fit_transform(dataset['baseVolume'].values.reshape(-1,1))
    model = generate_model('lstm_five_model.h5', 5, 4)
    values = data.values
    values = values.astype('float32')
    
    
    ascii_banner = pyfiglet.figlet_format("Five Minute Kraken !!!")
    print(ascii_banner)
    
    #Main Script Loop Starts Here
    while x <= loops:
        wait = 300 #seconds
        start = time.time() #Start our stopwatch for benchmarking purposes.
        
        #Call the Kraken! The Kraken API method returns a dictionary object with our ticker data.
        try:
            print('Collecting Ticker Data')
            dict_btc = await get_kraken("BTC/USD")
            print(f"Current close for BTC/USD: ${dict_btc['close']}")
        except:
            dict_btc = {}
            print(f"Kraken fail at {x}")
        
        #Gather and Analyze Tweets
        try:
            print("Gathering Tweets")
            twi_sent = await tweet_search_sent('bitcoin OR btc')
            print(f"Average compound Vader score of last 500 tweets about bitcoin {twi_sent}")
        except:
            twi_sent = 0
            print(f"Tweet fail at {x}")
        
        #Read and Analyze Reddit
        try:
            print("Reading Reddit")
            redd_sent = await get_redd_sent('bitcoin')
            print(f"Average compound Vader score of roughly last 100 reddit posts in r/bitcoin {redd_sent}")
        except:
            redd_sent = 0
            print(f"Reddit fail at {x}")
        #Delete Extraneous information from our ticker dictionary object.
        #'info' is a dictionary key pointing to a nested dictionary.
        try:
            del dict_btc['info']
        except:
            print("Empty Dict")
        
        if x > 7:
            offset = predicted_close - dict_btc['close']
            print(f"The difference between prediction and reality was: ${offset}")
            model.save_weights("Data/updated_lstm_five_model.h5")
       
        #Add our Sentiment Analysis columns. Append the dictionary object to our dataframe. Increment our loop counter x.
        dict_btc['twi_sent'] = twi_sent
        dict_btc['redd_sent'] = redd_sent
        df = df.append(dict_btc, ignore_index=True)
        x += 1
        
        if x > 5: #Once looped 5 five times, prepare set of 5 timesteps of new data for processing.
        
            older = pd.DataFrame()
            
            older = df.tail(5).copy()
            
            older['close'] = y_scaler.transform(older['close'].values.reshape(-1,1))
            older['baseVolume'] = vol_scaler.transform(older['baseVolume'].values.reshape(-1,1))
            older_scaled = await live_ticker_shaper(older)
            
            older_reframed = series_to_supervised(older_scaled, 5, 1)
            
            older_reframed.drop(older_reframed.columns[[21,22,23]], axis=1, inplace=True)
            
            older_values = older_reframed.values
            
        
        
        #""" This section currently isn't used but could be used in the future for Walk-Forward validation of the model.
        recent = pd.DataFrame()
        recent = recent.append(df.iloc[-1])
        recent['close'] = y_scaler.transform(recent['close'].values.reshape(-1,1))
        recent['baseVolume'] = vol_scaler.transform(recent['baseVolume'].values.reshape(-1,1))
        recent_scaled = await live_ticker_shaper(recent)
        recent_scaled = series_to_supervised(recent_scaled, 0, 1)
        recent_scaled = recent_scaled.values
        recent_scaled = recent_scaled.reshape(1, 1, 4)
        target = recent_scaled[:,0]
        #"""
        
        if x > 6: #Update Model Fit with new block of data.
            
            train_X, train_y = older_values[:, :20], older_values[:, 20:]
            
            train_X = train_X.reshape(5, 5, 4)
            await live_fitter(model,train_X,train_y)
            
            yhat = model.predict(train_X)
            
            prediction = y_scaler.inverse_transform(yhat)
            predicted_close = prediction[-1][-1]
            print(f"Predicted Price: ${prediction[-1][-1]}")
        
        #Printing Info from the Loop
        print(f"DataFrame Shape: {df.shape}")
        end = time.time()
        runtime = end - start ###https://www.blog.pythonlibrary.org/2016/05/24/python-101-an-intro-to-benchmarking-your-code/
        msg = "The runtime for this loop took {time} seconds to complete"
        print(msg.format(time=runtime))
        avg_time += runtime
        
        #Subtract the loop runtime from the wait time such that our data points are more evenly distributed.
        #And Print a countdown clock
        wait -= int(round(runtime,0))
        print("Remaining Wait: (ctrl+c to break) ")
        for i in range(wait): ###https://stackoverflow.com/questions/25189554/countdown-clock-0105/50148334
            mins, secs = divmod(wait-i, 60)
            timeformat = '{:02d}:{:02d}'.format(mins, secs)
            print(timeformat, end='\r')
            await asyncio.sleep(1)
            
    #Finally, calculate average loop runtime, save weights.
    avg_time = avg_time / x
    msg = "The average runtime for each loop was {time} seconds to complete"
    print(msg.format(time=avg_time))
    model.save_weights("Data/updated_lstm_five_model.h5")

#Call our main function.
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
