import matplotlib.pyplot as plot
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
import datetime as dt
import math
import numpy as np 
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Set the start and end date of the experiment 
start_ = dt.datetime(2014, 6, 8)
end_ = dt.datetime(2019, 6, 17)

# Function to calculate th True Range of the stock
def calc_true_range_(row):

    high_ = row['High']
    low_ = row['Low']
    open_ = row['Open']
    close_ = row['Close']
    volume_ = row['Volume']
    prev_close_ = row['Prev Close']

    max_1_ = high_ - low_
    max_2_ = abs(high_ - prev_close_)
    max_3_ = abs(low_- prev_close_)

    TR_ = max(max_1_,max_2_,max_3_)

    return TR_

# def calculate_parabolic_SAR(data):

#     lenght_df_ = len(data)
#     high_ = data['High']
#     low_ = data["Low"]
#     close_ = data["Close"]
#     open_ = data['Open']
#     df_psar_ = close_[0:len(close_)]
#     psarbull_ = [none] * lenght_df_
#     psarbear_ = [none] * lenght_df_
#     bull_ = True
#     iaf_ = .02
#     maxaf_ = .2
#     af = iaf
#     ep_ = low_[0]
#     hp_ = high_[0]
#     lp_ = low[0]

#     for i in range(2,lenght_df_):
#         if bull_:
#             df_psar_[i] = df_psar_[i - 1] + af * (hp - df_psar_[i - 1])



# Directory where the data (csvs) are, be sure to change to match your personal directory
dir_ = "C:\\Users\\137610\\OneDrive - Hitachi Consulting\\Core Connect\\Data"

# Global counter to sort through stock data files
file_counter_ = 1

# Change the working directory to where the data sits
os.chdir(dir_)

# Loop through the directory to ingest each file
for file_ in os.listdir(dir_):
    # Read the csv
    df_stocks_temp_ = pd.read_csv(file_)
    # Grab the ticker
    df_stocks_temp_['Ticker'] = file_.split(".")[0]
    # Populate the previous close
    df_stocks_temp_['Prev Close'] = df_stocks_temp_['Close'].shift(1)
    # Calculate the True Range for each row for the ATR calculation
    df_stocks_temp_['Prev Close'] = df_stocks_temp_['Prev Close'].fillna(method='bfill')
    df_stocks_temp_['TR'] = df_stocks_temp_.apply(calc_true_range_,axis=1)
    # Calculate the ATR
    df_stocks_temp_['ATR'] = df_stocks_temp_['TR'].ewm(span=14).mean()
    # Drop the TR column
    df_stocks_temp_.drop(['TR'], axis = 1)
    # Calculate RSI here 
    # Calculate the difference between the close prices
    df_delta_= df_stocks_temp_['Close'].diff()
    # Copy the delta to two different data frames
    df_dUp_, df_dDown_ = df_delta_.copy(),df_delta_.copy()
    # Find those deltas less than zero and drop then 
    df_dUp_[df_delta_<0] = 0
    # find those deltas greater than zerp and drop them
    df_dDown_[df_delta_>0] = 0

    roll_up_ = df_dUp_.rolling(14).mean()
    roll_down_ = df_dDown_.rolling(14).mean().abs()

    RS_ = roll_up_/roll_down_

    RSI_ = 100.0 - (100.0 / (1.0 + RS_))

    df_stocks_temp_['RSI'] = RSI_

    df_stocks_temp_['RSI'] = df_stocks_temp_['RSI'].fillna(method='bfill')

    # If the master data frame is being initialized then populate
    if file_counter_ == 1:
        df_stocks_ = df_stocks_temp_
    # Else append dataframe
    else:
        df_stocks_ = df_stocks_.append(df_stocks_temp_)
    # Clear temp dataframe
    df_stocks_temp_ = df_stocks_temp_.iloc[0:0]

    # Increment file count
    file_counter_ += 1

# Reset the dataframe index
df_stocks_ = df_stocks_.set_index(['Date','Ticker'])
# Sort dataframe by date
df_stocks_ = df_stocks_.sort_values('Date')

# Get all the unique tickers 
unique_stocks_ = df_stocks_.index.levels[1]

#df_stocks_.to_csv("Test.csv", index = True)

for stock_ in unique_stocks_:
    df_data_ = df_stocks_[df_stocks_.index.get_level_values(1) == stock_]
    df_data_ = df_data_.reset_index()
    df_data_['Date'] = pd.to_datetime(df_data_['Date'])
    df_data_['Ticker'] = df_data_['Ticker'].astype(str)
    ticker_ = str(df_data_['Ticker'][0])

    x_ = df_data_['ATR']
    y_ = df_data_['Close']

    fig, ax1_ = plot.subplots()
    ax1_.set_xlabel('Year')
    ax1_.set_ylabel('ATR')
    ax1_.plot(df_data_['Date'],df_data_['ATR'],color='red')
    ax1_.set_title(ticker_)

    ax2_ = ax1_.twinx()

    ax2_.set_ylabel('Close')
    ax2_.plot(df_data_['Date'],df_data_['Close'],color='blue')

    fig.tight_layout()
    plot.show()

    x_train_,x_test_,y_train_,y_test_ = train_test_split(x_,y_, test_size =.5)

    LinearRegressor_ = linear_model.LinearRegression()

    LinearRegressor_.fit(x_.values.reshape(-1,1),y_.values.reshape(-1,1))

    prediction_ = LinearRegressor_.predict(x_.values.reshape(-1,1))

    df_data_['Prediction_ATR'] = prediction_

    print(LinearRegressor_.score(x_.values.reshape(-1,1),y_.values.reshape(-1,1)))


    x_ = df_data_['RSI']
    y_ = df_data_['Close']

    fig, ax1_ = plot.subplots()
    ax1_.set_xlabel('Year')
    ax1_.set_ylabel('RSI')
    ax1_.plot(df_data_['Date'],df_data_['RSI'],color='red')
    ax1_.set_title(ticker_)

    ax2_ = ax1_.twinx()

    ax2_.set_ylabel('Close')
    ax2_.plot(df_data_['Date'],df_data_['Close'],color='blue')

    fig.tight_layout()
    plot.show()

    x_train_,x_test_,y_train_,y_test_ = train_test_split(x_,y_, test_size =.5)

    LinearRegressor_ = linear_model.LinearRegression()

    LinearRegressor_.fit(x_.values.reshape(-1,1),y_.values.reshape(-1,1))

    prediction_ = LinearRegressor_.predict(x_.values.reshape(-1,1))

    df_data_['Prediction_RSI'] = prediction_

    print(LinearRegressor_.score(x_.values.reshape(-1,1),y_.values.reshape(-1,1)))

    df_data_.to_csv(ticker_+"_output.csv", index = True)



    break

   



