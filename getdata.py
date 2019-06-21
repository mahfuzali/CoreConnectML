from pandas_datareader import data as web
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
import datetime as dt
import math
import numpy as np 
import pandas as pd
import os

# Set the start and end date of the experiment 
start_ = dt.datetime(2014, 6, 8)
end_ = dt.datetime(2019, 6, 17)

# Directory where the data (csvs) are, be sure to change to match your personal directory
dir_ = "C:\\Users\\137610\\OneDrive - Hitachi Consulting\\Core Connect\\Data"

# Global counter to sort through stock data files
file_counter_ = 1

# Change the working directory to where the data sits
os.chdir(dir_)

# Function to calculate the True Range in each row
def calc_tr_(row):

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

# Loop through the directory to ingest each file
for file_ in os.listdir(dir_):
    # Read the csv
    df_stocks_temp_ = pd.read_csv(file_)
    # Grab the ticker
    df_stocks_temp_['Ticker'] = file_.split(".")[0]
    # Populate the previous close
    df_stocks_temp_['Prev Close'] = df_stocks_temp_['Close'].shift(1)
    # Calculate the True Range for each row for the ATR calculation
    df_stocks_temp_['TR'] = df_stocks_temp_.apply(calc_tr_,axis=1)
    # Create increment for ATR calculation
    atr_counter_ = 1
    # Find the length of the TR column then find how many iterations of 14 days 
    length_atr_ = len(df_stocks_temp_['TR'])
    iterations_ = length_atr_%14

    # For loop to grab frames of TR data
    for i in range(0,iterations_):
        # Grab TR data in increments of 14
        df_temp_atr_ = df_stocks_temp_['TR'][(atr_counter_-1)*14:atr_counter_*14]
        # Length of values to iterate and calculate over
        length_df_temp_atr_ = len(df_temp_atr_)
        # Loop to Calculate ATR
        for j in range(0,length_df_temp_atr_):
            # Grab the True Range value
            ATR_temp_ = df_temp_atr_[((atr_counter_-1)*14)+j]
            # If the value is being initializated then populate
            if i == 0:
                ATR_ = ATR_temp_
            # If past initialization then add the TR values for the iterable period
            else:
                ATR_ = ATR_ + ATR_temp_
        # Populate the master data frame with the ATR values for the respective stock 
        df_stocks_temp_['ATR'] = ATR_
        # Increment ATR Counter 
        atr_counter_ += 1

    # Drop TR Column here

    
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


df = web.get_data_yahoo('AAPL', start=start, end=end)
#print(df.tail())

def prepare_data(df, forecast_col, forecast_out, test_size):
    label = df[forecast_col].shift(-forecast_out); #creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]]); #creating the feature array
    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing
    label.dropna(inplace=True); #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=test_size) #cross validation 

    response = [X_train, X_test , Y_train, Y_test, X_lately]
    return response


forecast_col = 'Adj Close'#choosing which column to forecast
forecast_out = 14 #how far to forecast 
test_size = 0.3; #the size of my test set

X_train, X_test, Y_train, Y_test , X_lately = prepare_data(df, forecast_col, forecast_out, test_size); #calling the method were the cross validation and data preperation is in

learner = linear_model.LinearRegression(); #initializing linear regression model

learner.fit(X_train,Y_train); #training the linear regression model
score = learner.score(X_test,Y_test); #testing the linear regression model

forecast = learner.predict(X_lately); #set that will contain the forecasted data


def workdays(d, count, excluded=(6, 7)):
    days = []
    numberofday = 0
    while numberofday < count:
        if d.isoweekday() not in excluded:
            days.append(d)
            numberofday += 1 
        d += dt.timedelta(days=1)
    return days

end_date = end + dt.timedelta(days=1)
list_date = workdays(end_date, 14)
date_list = [dt.datetime.strftime(i, "%Y-%m-%d") for i in list_date]

predicted = pd.DataFrame(forecast, date_list, columns =['Predicted Adj Close']) 

p_start = list_date[0]
p_end = list_date[len(list_date) - 1]
new_df = web.get_data_yahoo('AAPL', start=p_start, end=p_end)
print(new_df)

predicted['Actual Adj Close'] = new_df['Adj Close']
print(predicted)

