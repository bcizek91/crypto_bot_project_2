## Doge Pirates of the Cryptobean
***
![](Images/doge.jpg)
<br><br>

### **Team Members**
* Abdullahi Said 
* Matthew Newkirk
* James Reeves
* Blake Cizek
<br><br>

### **Project Description**
Develop a python script that leverages features from at least one Machine learning framework, signals derived from a crypto exchange, and sentiment from a media source. We will use Machine learning, sentiment analysis, and on-chain analysis to create a model that predicts the price of bitcoin and incorporate that into a python script that can be executed from a Terminal or Command prompt.
<br><br>

### **Objectives / Project Questions to Answer**
* Use machine learning to create a model to predict price movements of bitcoin
* If time allows expand to other crypto coins
* Implement our model into a trading bot/dcript
* If time allows, develop a dashboard for displaying useful feature set data.
<br><br>

### **Data Sources (APIs, Datasets)**
* Crypto asset information
    * CCXT Library (e.g. Kraken API)
* Sentiment Text
    * Twitter API
    * Reuters API
    * Reddit API
<br><br>

**Rough Breakdown of Tasks**
* Finding Data & Ingestion
* Data Preparation
* Feature Engineering
* Model Training
* Model Evaluation
* Trade-Strategy Execution
* Predictions and Conclusions
* Stretch Goals include additional crypto assets and data visualization features


### **Conclusion/ Final Results**
The model that we used for this problem was a kerras LSTM model. Long Short-Term Memory (LSTM) is a type of recurrent neural network that can learn the order dependence between items in a sequence. LSTMs have the promise of being able to learn the context required to make predictions in time series forecasting problems, rather than having this context pre-specified and fixed. The model is designed to consider the following features of BTC/USD data: Volume,  Circulating Supply, Supply Ratio, Volume Supply Ratio, MVRV ratio, NUPL ratio, five day rolling volatility, MarketCap supply ratio, and finally Closing price. The model receives these features as five inputted time lags cross referenced to one output time lag forecasting the next day’s closing price. The model implements a couple of dropout layers in-between hidden layers to prevent overfitting during the training process at 20%. 

The model has struggled to learn the rapid rise in BTC price towards the end of the dataset. Adding additional features such as rolling volatility and market supply cap ratio helped the model to grasp this sudden increase of the price. Application of repeated fittings of varying batch sizes (finally settling upon a batch size of 7, or a week) also gradually helped the LSTM to ‘learn’ the price surge. Repeated fittings of training data also helped lower the the RMSE to 5158 with a R2 of 0.79, and closing price at an average of 9000 dollars to around 5000. Additionally tweaking of both the feature set and model composition will likely lead to further improvements.

Though valuable as a learning exercise, formulating the machine learning forecast problem as a regression problem is potentially a bad fit for the proposed usage of the model. Redefining the problem as classification would potentially be a better fit for the purpose of combining ML forecasting with other more traditional ALGO trade logic signals. We believe the model would be better at predicting the general directional price movements(up or down) rather than the exact amount. Another recommendation could be to increase the training period to include recent price spikes.

