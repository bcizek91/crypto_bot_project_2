# Doge Pirates of the Cryptobean
___
![](Images/doge.jpg)
<br><br>
### **Team Members**
* Abdullahi Said 
* Matthew Newkirk
* James Reeves
* Blake Cizek
<br><br>

### **Project Description**
___
<ins>Learning Objectives:</ins>

* Develop domain knowledge of cryptocurrency exchanges and machine learning expertise as it applies to developing algorithmic trading strategies

<ins>Project Objectives:</ins>

1. Develop an analytical framework which utilizes a multivariate regression to predict the closing price of Bitcoin (BTC), sourced from various exchange metrics 
2. Wireframe a robo-advisor that will utilize derived price predictions and and social media sentiment to recommend the appropriate trading action & expand to other cryptocurrencies
3. Create the foundation for expanding to additional cryptocurrencies across exchanges

<ins>Summary:</ins>

* Utilizing leads and lags at the day-grain, trained an LSTM with partial success which was able to predict BTC closing price with a Root Mean Squared Error of 5158 and R2 of 0.79, struggling to scale with extreme volatility in the near term
* Wireframed the foundations of a trading bot at 5 min intervals utilizing social media sentiment within about $192 dollars on average.
<br><br>
### **Objectives / Project Questions to Answer**
___
<ins>Data Preparation & Feature Engineering:</ins>

Domains:
* Cryptocurrency Metrics (CoinMarketCap, Blockchain.com, Kraken/CCXT) from late 2013 to present day
    * OHLCV, Market Capitalization, Circulating Supply, Max Supply and calculated ratios
* On-chain Indicators (GlassNode)
    * NUPL and MVRV Z-ratio
* Social media sentiment metrics (Reddit & Twitter)** at 5 min intervals
    * VADER Scores

APIs:
* Kraken/CCXT, GlassNode, Reddit, Blockchain.com, Twitter

Dataframe Grain:
    * Day-grain with 5, 1-day lags
<br><br>
<ins>Model Preparation & Training:</ins>

Framework tested both locally on Jupyter Lab and on Google Collab:
Tensorflow Recurrent Neural Nets (RNNs) via Keras API, utilizing Long Short-Term Memory (LSTM) layers
<br><br>
** Did not have sufficient data access to cover training timeframe for average social media sentiment scores at the Day grain.
<br><br>
### **Breakdown of Tasks**
___
* Finding Data & Ingestion
* Data Preparation
* Feature Engineering
* Model Training
* Model Evaluation
* Predictions and Conclusions
* Stretch Goals included trade strategy indicators and working 5 min trading bot script
<br><br>
### **Conclusion/ Final Results**
___
The model that we used for this problem was a kerras LSTM model. Long Short-Term Memory (LSTM) is a type of recurrent neural network that can learn the order dependence between items in a sequence. LSTMs have the promise of being able to learn the context required to make predictions in time series forecasting problems, rather than having this context pre-specified and fixed. The model is designed to consider the following features of BTC/USD data: Volume,  Circulating Supply, Supply Ratio, Volume Supply Ratio, MVRV ratio, NUPL ratio, five day rolling volatility, MarketCap supply ratio, and finally Closing price. The model receives these features as five inputted time lags cross referenced to one output time lag forecasting the next day’s closing price. The model implements a couple of dropout layers in-between hidden layers to prevent overfitting during the training process at 20%. 

The model has struggled to learn the rapid rise in BTC price towards the end of the dataset. Adding additional features such as rolling volatility and market supply cap ratio helped the model to grasp this sudden increase of the price. Application of repeated fittings of varying batch sizes (finally settling upon a batch size of 7, or a week) also gradually helped the LSTM to ‘learn’ the price surge. Repeated fittings of training data also helped lower the the RMSE to 5158 with a R2 of 0.79, and closing price at an average of 9000 dollars to around 5000. Additionally tweaking of both the feature set and model composition will likely lead to further improvements.

Though valuable as a learning exercise, formulating the machine learning forecast problem as a regression problem is potentially a bad fit for the proposed usage of the model. Redefining the problem as classification would potentially be a better fit for the purpose of combining ML forecasting with other more traditional ALGO trade logic signals. We believe the model would be better at predicting the general directional price movements(up or down) rather than the exact amount. Another recommendation could be to increase the training period to include recent price spikes.

**Loss vs. Epochs**
![](Images/loss_chart.png)
**LSTM Prediction Results**
![](Images/lstm_prediction.png)

## 5 Min Kraken
___
* This script collects basic BTC ticker data every five minutes along with sentiment scores for recent posts about Bitcoin on both Twitter and r/Bitcoin. 
* It then fits data into a pre-trained LSTM with the same construction as used during our analysis. The LSTM then attempts to predict the next value for BTC’s closing price.
![](Images/5_min_kraken.png)
